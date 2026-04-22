const CACHE_NAME = 'kvcache-view-v2'
const urlsToCache = [
    './',
    './index.html',
    './inference.html',
    './train.html',
    './visualization.js',
    './manifest.json',
    './icon-192.png',
    './icon-512.png',
]

self.addEventListener('install', (event) => {
    self.skipWaiting()
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(urlsToCache)
        }),
    )
})

// Network-first for navigations/HTML so renamed or replaced pages don't get
// shadowed by a stale cache entry. Cache-first for other assets.
self.addEventListener('fetch', (event) => {
    const req = event.request
    const isHTML = req.mode === 'navigate' || (req.headers.get('accept') || '').includes('text/html')

    if (isHTML) {
        event.respondWith(
            fetch(req)
                .then((response) => {
                    const copy = response.clone()
                    caches.open(CACHE_NAME).then((cache) => cache.put(req, copy))
                    return response
                })
                .catch(() => caches.match(req).then((r) => r || caches.match('./index.html'))),
        )
        return
    }

    event.respondWith(
        caches.match(req).then((response) => {
            if (response) {
                return response
            }
            return fetch(req)
        }),
    )
})

self.addEventListener('activate', (event) => {
    const cacheWhitelist = [CACHE_NAME]
    event.waitUntil(
        caches
            .keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames.map((cacheName) => {
                        if (cacheWhitelist.indexOf(cacheName) === -1) {
                            return caches.delete(cacheName)
                        }
                    }),
                )
            })
            .then(() => self.clients.claim()),
    )
})
