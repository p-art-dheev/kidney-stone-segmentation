# Elsevier CAS single-column template provenance

Official source: https://ctan.org/pkg/els-cas-templates (version 2.4, 2024-05-04).
Downloaded from the CTAN mirror archive on 2026-10-13.

`cas-sc.cls`, `cas-common.sty` and `cas-model2-names.bst` are unchanged upstream
files. Their original Elsevier copyright and LPPL 1.3c notices remain intact.
These supporting files allow compiling `main.tex` with the actual CAS
single-column class. This is not `elsarticle`, a generic article approximation,
or a document exported from a word processor.

Local source-level customization in `main.tex` replaces the default
"Preprint submitted to Elsevier" footer with a course-project draft footer and
suppresses an empty ORCID footnote. It does not change the official class files.

Figure and result-table content is project-specific, with citations and measured
record provenance. See `docs/REPRODUCIBILITY.md` for build commands.
