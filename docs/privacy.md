# Privacy

Two separate things are described here, and they have different answers: **the application
you run**, and **this documentation site**.

## The application

`pdf-ocr` is software you run yourself. There is no hosted service, no account, and no
telemetry of any kind. Nobody but you sees the documents you convert.

Concretely, the application:

- **sends nothing anywhere.** There is no analytics, no crash reporting, no update check, and
  no outbound network call. The web page loads no third-party assets — Tailwind is vendored
  into the repository, and the Content-Security-Policy is same-origin.
- **stores your files on your own disk**, under `UPLOAD_FOLDER`. The uploaded PDF is deleted
  as soon as the conversion ends, whether it succeeded, failed or was cancelled. The
  converted file stays until you download and discard it, or until cleanup removes it — files
  older than 24 hours are deleted, and task records an hour after their last update.
- **stores no personal data of its own.** The session cookie holds a list of conversion ids
  you started, nothing else. There is no identity to store.
- **writes logs** containing filenames, chosen options and errors. They go to stdout by
  default, and to a file only if you set `LOG_FILE`. They are yours.

One caveat worth being explicit about: some OCR engines download **language models** on first
use. EasyOCR and PaddleOCR fetch these from their own project infrastructure. That is a
download of model data, not an upload — your documents are never involved — but it is an
outbound connection, and it is theirs rather than ours. Tesseract, the default, does not do
this: its language packs come from your system package manager.

## This documentation site

These pages are static files served by **GitHub Pages**. There is no analytics script, no
cookie set by this site, and no tracking of any kind in the pages themselves.

GitHub serves them, and as the host it processes request data — including IP addresses — for
delivery and abuse prevention. That is governed by
[GitHub's Privacy Statement](https://docs.github.com/en/site-policy/privacy-policies/github-privacy-statement),
not by this project. If that matters to you, the entire site can be read from the
[repository](https://github.com/fabriziosalmi/pdf-ocr/tree/main/docs) or built locally.

The site's search runs entirely in your browser; queries are not sent anywhere.

## Contact

For anything about this project, including privacy questions, open an issue or email
`fabrizio.salmi@gmail.com`. For a suspected vulnerability, use the private channel described
on the [Security](/security) page instead.
