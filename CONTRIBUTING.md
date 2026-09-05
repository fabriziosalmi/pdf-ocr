# Contributing

Thanks for looking. This is a spare-time project, so responses may be slow, but issues and
pull requests are welcome.

## Before you open a pull request

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check .                    # lint
python -m unittest test_app -v  # tests
```

CI runs the same two on Python 3.11 to 3.14, plus a Docker job that builds the image, waits
for its HEALTHCHECK and asserts the container is not running as root, plus CodeQL. The first
three must pass before a pull request can merge.

Install Poppler and Tesseract if you can — without them, the tests that exercise real
rendering and real OCR skip silently rather than fail, so you get a green run that proved
less than it looks.

```bash
brew install poppler tesseract                          # macOS
sudo apt-get install -y poppler-utils tesseract-ocr      # Debian/Ubuntu
```

## What this project will and will not take

The [README](README.md#what-it-does-not-do) lists what has been decided against, with the
reasoning. Advanced preprocessing needing OpenCV, and built-in authentication or rate
limiting, are the two big ones. Please open an issue before building either — you would be
arguing against a decision, which is fine, but do it before writing the code.

## House rules that are not negotiable

These exist because each one was a real defect:

- **A control in the UI must be read by the server.** Ten form fields were once submitted and
  silently discarded. Do not add a checkbox that nothing consumes; remove it instead.
- **The OCR clean-up pass must never rewrite a character the engine produced.** It once
  applied `0`→`O`, `1`→`I`, `5`→`S` globally, corrupting every number in every document.
  `test_fix_common_ocr_errors_never_corrupts_content` guards this deliberately.
- **A green CI run is not proof.** Optional OCR engines are lazily imported and are not
  installed in CI, so a change can break one while every check stays green. Say so in the
  pull request if your change touches that path.
- **Do not put a line count, or any other self-invalidating number, in the documentation.**

## Commit messages

Explain why, not what — the diff already says what. If you are fixing a defect, say what the
broken behaviour was, so the next person reading `git log` understands the reason the code
looks the way it does.

## Security

Do not open a public issue for a vulnerability. See [SECURITY.md](SECURITY.md).

## Code of Conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
