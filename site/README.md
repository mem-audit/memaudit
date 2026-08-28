# memaudit landing page

Single self-contained `index.html` (inline CSS/JS, no build step). Only external dependency is Google Fonts; everything else works offline.

## Deploy to GitHub Pages

1. Repo → **Settings → Pages → Source: GitHub Actions**, then add the standard "Static HTML" workflow with `path: site` (the branch-based Pages option only serves `/` or `/docs`, not `/site`).
2. Push to `main`; the site goes live at `https://<owner>.github.io/memaudit/`. GitHub links in `index.html` point at the product org [`mem-audit/memaudit`](https://github.com/mem-audit/memaudit); the design-partner mailto points to `ansh.singh.160305@gmail.com`. The Hugging Face org is live at [https://huggingface.co/memaudit](https://huggingface.co/memaudit).

## Live deployment

- **URL:** https://ansh200516.github.io/memaudit-site/
- **Hosted on:** GitHub Pages, from the standalone public repo [`ansh200516/memaudit-site`](https://github.com/ansh200516/memaudit-site) — branch `main`, path `/` (plain branch-based Pages; no Actions workflow needed since `index.html` sits at the repo root). The repo lives under the personal account for now and can later transfer to the product org [`mem-audit`](https://github.com/mem-audit) (Pages URL then becomes `https://mem-audit.github.io/memaudit-site/`). The Actions-based setup described above still applies to the eventual main product repo.
- **Deployed & verified:** 2026-08-28 — hero is the powered TinyLlama n=100/200 run (TPR 0.180, CI [0.110, 0.269]); HTTP 200 after Pages rebuild.
- **Redeploy after editing `site/index.html`** (from the main repo root):

  ```sh
  git clone https://github.com/ansh200516/memaudit-site.git /tmp/memaudit-site-deploy  # skip if checkout already exists
  cp site/index.html /tmp/memaudit-site-deploy/index.html
  cp site/case-study.html /tmp/memaudit-site-deploy/case-study.html
  git -C /tmp/memaudit-site-deploy add index.html case-study.html
  git -C /tmp/memaudit-site-deploy commit -m "Update landing page and flagship case study"
  git -C /tmp/memaudit-site-deploy push
  ```

  Pages rebuilds automatically on push (typically live in under a minute).
