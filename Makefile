# KV Cache Visualization Makefile

.PHONY: serve run stop clean help check-commits fix-commits format check-format

# Default port
PORT ?= 8000

# Python executable
PYTHON := python3

## serve: Start the web server
serve: stop
	@echo "Starting server..."
	@sh -c '$(PYTHON) -m http.server $(PORT) > server.log 2>&1 & echo $$! > server.pid'
	@sleep 1
	@echo ""
	@echo "✨ KV Cache Visualization running!"
	@echo "📍 URL: http://localhost:$(PORT)"
	@echo ""
	@echo "Use 'make stop' to stop the server"

## run: Alias for serve
run: serve

## stop: Stop the web server
stop:
	@sh -c 'test -f server.pid && kill $$(cat server.pid) 2>/dev/null || true'
	@rm -f server.pid

## clean: Clean up server files
clean: stop
	@rm -f server.log server.pid
	@echo "Cleaned up server files"

## check-commits: Check if commit messages follow CLAUDE.md guidelines
check-commits:
	@echo "Checking commit messages for CLAUDE.md compliance..."
	@failed=0; \
	for commit in $$(git log --format=%H origin/main..HEAD 2>/dev/null || git log --format=%H -10); do \
		msg=$$(git log -1 --format=%B $$commit); \
		subject=$$(echo "$$msg" | head -1); \
		body=$$(echo "$$msg" | tail -n +2); \
		errors=""; \
		if echo "$$subject" | grep -q "Generated-by:\|Signed-off-by:\|Co-Authored-By:"; then \
			errors="$$errors\n  ❌ Subject line contains attribution (should be in body only)"; \
		fi; \
		if ! echo "$$body" | grep -q "^Generated-by: "; then \
			errors="$$errors\n  ❌ Missing 'Generated-by:' tag in body (e.g. 'Generated-by: Claude AI')"; \
		fi; \
		if ! echo "$$body" | grep -q "^Signed-off-by: "; then \
			errors="$$errors\n  ❌ Missing 'Signed-off-by:' tag in body"; \
		fi; \
		if echo "$$body" | grep -q "Co-Authored-By:"; then \
			errors="$$errors\n  ❌ Contains incorrect 'Co-Authored-By' (use 'Generated-by' instead)"; \
		fi; \
		if echo "$$body" | grep -q "Claude Code"; then \
			errors="$$errors\n  ❌ Contains 'Claude Code' reference (use 'Claude AI' instead)"; \
		fi; \
		if [ ! -z "$$errors" ]; then \
			echo ""; \
			echo "❌ Commit $$commit has issues:"; \
			echo "  Subject: $$subject"; \
			printf "$$errors\n"; \
			failed=1; \
		fi; \
	done; \
	if [ $$failed -eq 0 ]; then \
		echo "✅ All commits follow CLAUDE.md guidelines!"; \
	else \
		echo ""; \
		echo "⚠️  Fix with: make fix-commits"; \
		exit 1; \
	fi

## fix-commits: Automatically fix commit messages to follow CLAUDE.md guidelines
fix-commits:
	@echo "Fixing commit messages to comply with CLAUDE.md..."
	@echo "⚠️  This will rewrite git history!"
	@read -p "Continue? (y/N) " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		SOB_NAME=$$(git config user.name); \
		SOB_EMAIL=$$(git config user.email); \
		git filter-branch -f --msg-filter ' \
			msg=$$(cat); \
			subject=$$(echo "$$msg" | head -1 | sed "s/ Generated-by:.*//"); \
			body=$$(echo "$$msg" | tail -n +2); \
			cleaned_body=$$(echo "$$body" | grep -v "^Generated-by:" | grep -v "^Signed-off-by:" | grep -v "^Co-Authored-By:"); \
			echo "$$subject"; \
			if [ ! -z "$$cleaned_body" ]; then \
				echo "$$cleaned_body"; \
			fi; \
			echo ""; \
			echo "Generated-by: Claude AI"; \
			echo "Signed-off-by: '"$$SOB_NAME"' <'"$$SOB_EMAIL"'>" \
		' -- --all; \
		echo "✅ Commit messages fixed!"; \
		echo "⚠️  You may need to force push: git push --force-with-lease"; \
	else \
		echo "Aborted."; \
	fi

## format: Format HTML and JS files with prettier
format:
	@if command -v prettier > /dev/null 2>&1; then \
		echo "Formatting HTML and JS files with Prettier..."; \
		prettier --write "*.html" "*.js"; \
		echo "✅ Files formatted!"; \
	elif command -v npx > /dev/null 2>&1; then \
		echo "Using npx to run Prettier..."; \
		npx prettier --write "*.html" "*.js"; \
		echo "✅ Files formatted!"; \
	else \
		echo "⚠️  Prettier not found."; \
		echo ""; \
		echo "Install options:"; \
		echo "  1. npm install        (local installation)"; \
		echo "  2. npm install -g prettier  (global installation)"; \
		echo ""; \
		exit 1; \
	fi

## check-format: Check if HTML and JS files are properly formatted
check-format:
	@if command -v prettier > /dev/null 2>&1; then \
		echo "Checking code formatting..."; \
		prettier --check "*.html" "*.js" || \
		(echo "❌ Code formatting issues found. Run 'make format' to fix."; exit 1); \
		echo "✅ All files are properly formatted!"; \
	elif command -v npx > /dev/null 2>&1; then \
		echo "Using npx to check formatting..."; \
		npx prettier --check "*.html" "*.js" || \
		(echo "❌ Code formatting issues found. Run 'make format' to fix."; exit 1); \
		echo "✅ All files are properly formatted!"; \
	else \
		echo "⚠️  Prettier not found."; \
		echo ""; \
		echo "Install options:"; \
		echo "  1. npm install        (local installation)"; \
		echo "  2. npm install -g prettier  (global installation)"; \
		echo ""; \
		exit 1; \
	fi

## help: Show this help message
help:
	@echo "KV Cache Visualization - Memory Growth Demo"
	@echo "==========================================="
	@echo ""
	@echo "Server Commands:"
	@echo "  make              Start server at http://localhost:8000"
	@echo "  make serve        Start server"
	@echo "  make run          Same as serve"
	@echo "  make stop         Stop the running server"
	@echo "  make clean        Stop server and clean up files"
	@echo ""
	@echo "Code Quality:"
	@echo "  make check-commits   Check if commit messages follow CLAUDE.md"
	@echo "  make fix-commits     Fix commit messages to follow CLAUDE.md"
	@echo "  make format          Format HTML/JS files with Prettier"
	@echo "  make check-format    Check if HTML/JS files are formatted"
	@echo ""
	@echo "  make help         Show this help message"
	@echo ""
	@echo "Options:"
	@echo "  PORT=8080 make serve    Use custom port (default: 8000)"
	@echo ""
	@echo "Features:"
	@echo "  • Watch KV cache memory explode from 0 to 100M tokens"
	@echo "  • Compare 5 models from 1B to 671B parameters"
	@echo "  • Switch data types (FP32/FP16/BF16/INT8/INT4)"
	@echo "  • Speed controls from 0.5x to 100x"
	@echo ""
	@echo "The visualization demonstrates why LMCache optimizations are critical"
	@echo "as context lengths grow to millions of tokens."

# Default target
.DEFAULT_GOAL := serve
