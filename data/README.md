# Data

## `samples/`

The `sample_<n>.csv` files. Not versioned: download the content of `data/samples` from the
DALT-ANON-STUDY repository.

### Download `data/samples/`

Run from the root of this repository (only `data/samples` is fetched, not the full history of
DALT-ANON-STUDY):

```bash
REPO_URL=https://github.com/VictorMenestrel/DALT-for-Speech-Anonymizers 
TMP_DIR=$(mktemp -d)
git clone --depth 1 --filter=blob:none --sparse "$REPO_URL" "$TMP_DIR"
git -C "$TMP_DIR" sparse-checkout set data/samples
mkdir -p data/samples
cp -r "$TMP_DIR"/data/samples/. data/samples/
rm -rf "$TMP_DIR"
```
