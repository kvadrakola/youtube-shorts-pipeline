# Third-party notices

## Fonts (`pipeline/assets/fonts/`)

Bundled font files are **not** MIT-licensed. Each family is under the **SIL Open Font License 1.1**. Read the corresponding `*-OFL.txt` or `OFL.txt` in that directory before redistributing or modifying fonts.

## Python dependencies

Runtime libraries are declared in `pipeline/requirements.txt` and are subject to their respective upstream licenses when you install them. This repository does not ship those packages as source.

## Outro HTML template

The default card under `poster/outro_card_template/` uses Tailwind CSS and Google Fonts via CDN at render time. Network access is required when Playwright loads the template unless you self-host those assets.
