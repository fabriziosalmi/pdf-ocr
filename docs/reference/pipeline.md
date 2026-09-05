# How a conversion works

Useful when something goes wrong, because almost every failure belongs to exactly one of
these stages.

## 1. Upload

The request is checked before anything touches disk:

- `.pdf` extension **and** the `%PDF-` magic bytes. The extension alone says nothing about
  content, and the file is about to be handed to Poppler.
- Engine, output format and language against allowlists. The output format used to be
  interpolated straight into the output path.
- Size against `MAX_UPLOAD_MB`, enforced by Flask before the body is fully read.

A UUID is generated, the file is saved as `<uuid>_<sanitised name>.pdf`, and the id is added
to the session's list of owned conversions.

## 2. Page count

`pdfinfo` reports the page count. This is required rather than best-effort: it decides how
the work is batched, and without it the whole document would have to be rendered just to
find out how long it is. Over `MAX_PAGES`, the conversion is rejected here.

## 3. Render and OCR, in batches

The loop is the heart of it:

```
for each batch of RENDER_BATCH_SIZE pages:
    check for cancellation
    render those pages with Poppler at 300 or 600 DPI
    for each page:
        write the PNG, preprocess if asked
        run the OCR engine
        delete the PNG
        update progress
        check for cancellation
```

Batching is what keeps memory bounded. Rendering the whole document at once — which it used
to do — is several gigabytes for a 100-page PDF at 600 DPI, well over a typical container
limit.

Cancellation is checked twice per batch because a single 600 DPI page can take a while.

## 4. Clean-up pass

Applied to each page's text. It **only** adjusts layout:

- re-joins words hyphenated across a line break (`exam-\nple` → `example`)
- removes spaces before closing punctuation
- strips trailing whitespace
- collapses runs of blank lines

It never rewrites a character the engine produced. An earlier version substituted `0`→`O`,
`1`→`I` and `5`→`S` globally, corrupting every number in every document; a regression test
pins those specific substitutions as forbidden.

Paragraph reflow — turning single newlines into spaces — is off by default, because it
destroys tables, addresses and invoices.

## 5. Assembly

| Format | What is written |
|---|---|
| DOCX | One paragraph per page's text, with a page break between pages. |
| TXT | Text with `--- Page Break ---` between pages. |
| Markdown | Paragraphs split on blank lines, `---` between pages. |
| HTML | Each paragraph in `<p>`, entities escaped, `<hr class="page-break">` between pages. |

The uploaded PDF is then deleted, whatever the outcome — success, failure or cancellation.

## Where the state lives

One JSON file per conversion under `<UPLOAD_FOLDER>/.tasks/`, written with a
write-then-rename so a reader never sees a partial file.

This is what lets more than one gunicorn worker serve the same conversion: the upload lands
on one worker and the status poll on another, and both read the same file. It also survives
a restart.

Cancellation uses a **separate** marker file rather than a field in that record. Updating the
record is a read-modify-write with no locking, so a flag stored there would race with the
per-page progress writes — the worker could read the record, the cancel could land, and the
worker would then write its stale copy back, losing the cancellation with no trace.

## Failure states

| State | Meaning |
|---|---|
| `processing` | Running. `progress` and `step` say where. |
| `completed` | Output written and downloadable. |
| `failed` | Carries an `error`. |
| `cancelled` | Stopped on request. Carries no error, because it is not one. |

A conversion whose worker dies never reaches a terminal state on its own, so a record that
has not progressed for `STALE_TASK_TIMEOUT` is reported as failed rather than left pending
forever.
