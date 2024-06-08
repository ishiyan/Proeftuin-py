# MICs

## Data sources

Sources for data used in `countries*.csv` and `ISO10383_MIC.*.csv` files:

- [ISO 10383 Market Identifier Codes](https://www.iso20022.org/market-identifier-codes)
- [csv](https://www.iso20022.org/sites/default/files/ISO10383_MIC/ISO10383_MIC.csv)
- [wikipedia](https://en.wikipedia.org/wiki/ISO_4217)
- [ISO 3166 country codes](https://www.iso.org/iso-3166-country-codes.html)

## Generating source code

- Download the latest data release from [ISO 10383 Market Identifier Codes](https://www.iso20022.org/market-identifier-codes).
  The direct link is [csv](https://www.iso20022.org/sites/default/files/ISO10383_MIC/ISO10383_MIC.csv).
- Rename the file from `ISO10383_MIC.csv` to `ISO10383_MIC.XX-YYY-20ZZ.csv` where `XX-YYY-20ZZ` is the data release date.
- Update the data release date in the `generate_mics.go` code.
- Copy `contries.all.csv` to the `contries.csv` to have all countries for the first run.
- Execute `go run generate_mics_python` in this folder.
- Check the generated `mics.py` file.
- If needed, `contries.selected.csv` to the `contries.csv` and repeat the run.
