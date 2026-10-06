# Audio

Audio evaluated by the ASR models.

| Folder | Content |
| --- | --- |
| `samples/` | Audio **without carrier phrase**: the isolated words partialy rated by the participants of the listening test, one folder per condition (`clean/`, `B2_mcadams/`, `B3_sttts/`, `B4_nac/`, `B5_asr_bn/`, `TTS_Qwen/`). Not versioned: download the content of `data/audio/samples` from the DALT-ANON-STUDY repository |
| `samples/your_anonymizer/` | The audio of your own anonymization system, for `src/take_the_test.py` (any folder name works) |
| `level_normalized_26dbfs_w_concat_carrier_phrase_2/` | Audio **with carrier phrase**: each target word is preceded by 5 other words of the same speaker, normalized to -26 dBFS, one folder per condition. The carrier words and the start time of the target word are given in `clean/English/test-data/concat_metadata.csv` |
| `TTS_Qwen/` | Synthetic speech of the target words **with carrier phrase**: "Please select the word &lt;word&gt;". They are also level normalized to -26 dBFS. |

### Download `data/audio/samples/`

Run from the root of this repository (only `data/audio/samples` is fetched, not the full
history of DALT-ANON-STUDY):

```bash
REPO_URL=https://github.com/VictorMenestrel/DALT-for-Speech-Anonymizers
TMP_DIR=$(mktemp -d)
git clone --depth 1 --filter=blob:none --sparse "$REPO_URL" "$TMP_DIR"
git -C "$TMP_DIR" sparse-checkout set data/audio/samples
mkdir -p data/audio/samples
cp -r "$TMP_DIR"/data/audio/samples/. data/audio/samples/
rm -rf "$TMP_DIR"
```

`src/run_eval.py` reads `samples/` by default and
`level_normalized_26dbfs_w_concat_carrier_phrase_2/` with `--carrier-phrase`. The files of a
condition keep the names of the `clean/` recordings (`<word>_<hash>.wav`), and
`samples/clean/English/test_design.csv` gives the DALT trial of each recording.

## License and citation

The audio files are not covered by the MIT license of this repository.

**`clean/` and the anonymized folders** (`B2_mcadams/`, `B3_sttts/`, `B4_nac/`, `B5_asr_bn/`,
in `samples/` and in `level_normalized_26dbfs_w_concat_carrier_phrase_2/`) come from, or are
derived from, the recordings of the
[cisco/multilingual-speech-testing](https://github.com/cisco/multilingual-speech-testing/tree/main/speech-intelligibility-DRT)
repository. They are licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The anonymized folders and
the versions with carrier phrase are modified versions of the original recordings.

The original recordings come from:

> Lechler, L., & Wojcicki, K. (2024). Crowdsourced Multilingual Speech Intelligibility Testing.
> *ICASSP 2024*, 1441–1445. https://doi.org/10.1109/ICASSP48485.2024.10447869

**`TTS_Qwen/`** (both the version in `samples/` and the one with carrier phrase) contains no
Cisco recordings. It was generated with the
[Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
model (voice "Ryan"), which is released under the
[Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0). The model is described in:

> Hu, H., Zhu, X., He, T., Guo, D., Zhang, B., Wang, X., Guo, Z., Jiang, Z., Hao, H., Guo, Z.,
> Zhang, X., Zhang, P., Yang, B., Xu, J., Zhou, J., & Lin, J. (2026). Qwen3-TTS Technical Report.
> *arXiv preprint arXiv:2601.15621*.

**`samples/your_anonymizer/`** holds your own files, under your own terms.
