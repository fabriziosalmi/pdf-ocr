## What this changes, and why

<!-- The diff says what. Say why — what was wrong, or what was missing. -->

## How you verified it

<!-- "CI is green" is a start, not an answer. Optional OCR engines are lazily
     imported and not installed in CI, so a change can break one while every
     check passes. If you ran the app, say so and say what you saw. -->

- [ ] `ruff check .`
- [ ] `python -m unittest test_app`
- [ ] Poppler and Tesseract installed locally, so the real-rendering and
      real-OCR tests ran instead of skipping
- [ ] Ran the app and exercised the change by hand

## Anything the tests do not cover

<!-- Say it plainly rather than leaving it to be discovered. -->
