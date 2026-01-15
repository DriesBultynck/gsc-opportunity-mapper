# GSC Opportunity Mapper

Upload two Google Search Console CSV exports:
- Queries export
- Pages export

The app clusters queries, maps clusters to pages, estimates CTR opportunity, and generates reporting + a `gpt_brief.json` file for use with a Custom GPT.

## Run locally
pip install -r requirements.txt
streamlit run app.py
