# GSC Opportunity Mapper (Stratified)

A sophisticated Streamlit application that analyzes Google Search Console (GSC) data to identify SEO opportunities through intelligent query clustering, intent classification, and opportunity scoring. Uses a **stratified clustering approach** that segments queries by brand status and search intent before clustering, resulting in more meaningful and actionable insights.

## Overview

This tool processes GSC query and page data to:
- **Cluster semantically similar queries** within brand/intent segments
- **Map query clusters to pages** using TF-IDF similarity
- **Calculate opportunity clicks** based on position-based CTR expectations
- **Prioritize opportunities** using a composite scoring system
- **Detect cannibalization risks** when multiple pages compete for the same queries
- **Generate actionable recommendations** for each cluster
- **Export comprehensive reports** including GPT-ready briefs

## Key Features

### 🎯 Stratified Clustering
Queries are segmented by:
1. **Brand Status**: Branded vs Non-Branded (based on configurable brand terms)
2. **Search Intent**: Navigational, Transactional, Commercial, or Informational

Clustering happens *within* each segment independently, ensuring more coherent topic groups.

### 📊 Opportunity Calculation
- Estimates potential clicks based on position-based CTR floors
- Uses median CTR per position bucket from your data (with industry fallbacks)
- Calculates opportunity as: `Impressions × (Expected CTR - Actual CTR)`

### 🎨 Priority Scoring
Composite score balancing:
- Current clicks (1.0× weight)
- Opportunity clicks (1.2× weight) 
- Total impressions (0.4× weight)

Results are banded into:
- **P1 (High)**: Top 10% of opportunities
- **P2 (Medium)**: Next 20%
- **P3 (Low)**: Bottom 70%

### 🔍 Cannibalization Detection
Identifies when multiple pages compete for the same query clusters (match scores within 3% of each other), flagging potential internal competition issues. Includes dedicated analysis table showing competing clusters and pages.

### 🌍 Multi-Language Support
Supports GSC exports in any language with automatic column detection:
- **Auto-detection**: Automatically detects column names in English, Dutch (including "vertoningen" for impressions), French, Spanish, German, and more
- **Manual override**: Easily map columns manually if auto-detection doesn't match your CSV
- **Visual mapping UI**: Expandable sections show detected columns and allow you to map each required field
- **Validation**: Ensures all required columns are mapped before processing begins

### 🎛️ Intent Keywords Management
- **Customizable intent classification**: Edit keywords used for each intent type (navigational, transactional, commercial, informational)
- **JSON export/import**: Save and load intent keyword configurations
- **Per-intent editing**: Edit keywords directly in the app or via the Intent Keywords tab

### 🔗 Singular/Plural Cluster Merging
- **Automatic detection**: Identifies clusters with singular/plural topic variations
- **Smart merging**: Merges clusters if they share the same intent and have singular/plural relationships
- **Optional feature**: Enable/disable via checkbox in settings

### 🤖 Optional Semantic Clustering
- **Sentence-transformers support**: Use semantic embeddings for better clustering (optional)
- **Better semantic understanding**: Captures synonyms, context, and meaning better than TF-IDF
- **Fallback**: Automatically falls back to TF-IDF if sentence-transformers is not available
- **Model**: Uses `all-MiniLM-L6-v2` by default (fast and efficient)

### 📊 Advanced Analysis Tables
- **All Keywords**: Filterable keyword-level table with cascading filters (Brand → Topic → Intent)
- **Cannibalization Risks**: Detailed analysis of competing clusters and pages
- **Action Priority**: Clusters sorted by effort level (low to high) for prioritization
- **Raw Data**: Original aggregated tables for reference

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd projects/gsc-opportunity-mapper
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run gsc_mapper_improved.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage

### Step 1: Export Data from Google Search Console

⚠️ **Important**: Google Search Console UI limits exports to **1,000 rows maximum**. For larger datasets, use the GSC API or export multiple date ranges.

Export two CSV files from GSC (same date range):

1. **Queries Export** (`Queries` report):
   - Required data: Query/Page column, Clicks, Impressions, CTR, Position
   - **Note**: Column names can be in any language (English, Dutch, French, Spanish, German, etc.)
   - Date range: Recommended 3-6 months for better clustering
   - **Limit**: Maximum 1,000 rows per export

2. **Pages Export** (`Pages` report):
   - Required data: Page/URL column, Clicks, Impressions, CTR, Position
   - Same date range as queries
   - **Note**: Column names can be in any language
   - **Limit**: Maximum 1,000 rows per export

### Step 2: Upload and Map Columns

1. Upload both CSV files using the file uploaders
2. **Map columns** (if needed):
   - Column mapping sections are expanded by default
   - Review auto-detected column mappings (marked with ✅)
   - Manually adjust any mappings using the dropdown menus
   - All required fields must be mapped before processing
3. The app will validate mappings and show errors if any are missing

### Step 3: Configure Settings

Configure all settings before starting clustering:

- **Client name**: Used in reports and briefs
- **Brand terms**: Comma-separated list of brand terms (e.g., `"acme, acme corp, acme inc"`)
  - Used to classify queries as branded/non-branded
  - Also included in navigational intent detection

**Clustering Settings:**
- **Use Semantic Clustering**: Optional - uses sentence-transformers for better semantic understanding (requires `sentence-transformers` library)
- **Enable Singular/Plural Merging**: Merges clusters with singular/plural variations if they share the same intent

**Intent Keywords:**
- Configure keywords used for intent classification per intent type
- Edit keywords directly in expandable sections
- Keywords are saved and used for processing

### Step 4: Start Clustering

1. Click the **"🚀 Start Clustering"** button to begin processing
2. Processing runs the full pipeline:
   - Query preprocessing and intent classification
   - Stratified clustering (by brand and intent)
   - Cluster labeling and opportunity calculation
   - Page mapping and cannibalization detection
3. Results are stored in session state for review

### Step 5: Review Results

Review results in four tabs:
- **Visuals**: Interactive bubble charts with cascading filters (Brand → Intent)
- **Tables**: Advanced analysis tables (All Keywords, Cannibalization Risks, Action Priority, Raw Data)
- **Intent Keywords**: Manage and export intent keyword configurations
- **Download**: ZIP report with all outputs

## Output Files

The downloadable ZIP report contains:

### `gpt_brief.json`
Structured JSON brief with:
- Top 20 pages by opportunity clicks
- Top 20 clusters with recommendations
- Metadata (client name, brand terms, generation timestamp)

### `gpt_brief.md`
Markdown version of the brief with formatted tables.

### `clusters.csv`
Detailed cluster data including:
- `cluster_id`: Unique cluster identifier
- `topic_label`: Auto-generated topic name
- `segment`: Brand status + Intent (e.g., "Non-Branded - transactional")
- `queries`: Number of queries in cluster
- `clicks`, `impressions`: Aggregated metrics
- `avg_position`: Average position
- `opportunity_clicks`: Estimated opportunity
- `priority_score`, `priority_band`: Scoring
- `primary_page`: Best matching page
- `match_score`: Similarity score (0-1)
- `runner_up_page`, `runner_up_score`: Second-best match
- `cannibalisation_risk`: Boolean flag
- `recommended_action`: Actionable recommendation
- `example_queries`: Top 5 queries in cluster

### `pages_segment.csv`
Page opportunities broken down by segment:
- Page URL and slug
- Segment (brand/intent combination)
- Impressions, clicks, opportunity clicks per segment
- Average position
- Page-level CTR metrics

### `top_pages_chart.png`
Static visualization showing top 20 pages with actual vs opportunity clicks.

## Methodology

### Intent Classification

Uses regex-based pattern matching with default rules:

- **Navigational**: Brand terms, login, contact, customer service
- **Transactional**: buy, order, pricing, quote, book, hire, near me
- **Commercial**: best, top, review, vs, compare, alternative
- **Informational**: how, what, why, guide, tutorial, definition

Default classification is "informational" if no patterns match.

### Clustering Algorithm

**For segments with < 10,000 queries:**
- Uses **Agglomerative Clustering** with cosine similarity
- Distance threshold: 0.75 (75% similarity required)
- Linkage: average

**Optional Semantic Clustering (sentence-transformers):**
- Uses pre-trained transformer models for semantic embeddings
- Better understanding of synonyms and context
- Same distance threshold (0.75) and clustering logic
- Falls back to TF-IDF if sentence-transformers is not available

**For segments with ≥ 10,000 queries:**
- Uses **MiniBatchKMeans** for performance
- Number of clusters: `max(5, queries / 10)`
- Batch size: 1024

**TF-IDF Vectorization:**
- N-gram range: (1, 3) - unigrams, bigrams, trigrams
- Min document frequency: 1
- Max document frequency: 0.95

### CTR Floors

Position-based CTR floors (industry baselines):
- Position 1-3: 15%
- Position 4-6: 8%
- Position 7-10: 4%
- Position 11-20: 1.5%
- Position 21+: 0.5%

Expected CTR uses the **higher** of:
1. Median CTR for that position bucket in your data
2. The floor value above

### Opportunity Calculation

```
Opportunity Clicks = Impressions × max(0, Expected CTR - Actual CTR)
```

Where Expected CTR is calculated per position bucket using the methodology above.

### Priority Score Formula

```
Priority Score = 1.0 × ln(1 + Clicks) 
               + 1.2 × ln(1 + Opportunity Clicks) 
               + 0.4 × ln(1 + Impressions)
```

Uses log scaling to balance high-volume vs high-potential opportunities.

### Page Mapping

Clusters are matched to pages using:
1. TF-IDF vectorization of cluster text (topic + intent + top queries)
2. TF-IDF vectorization of page URLs (tokenized paths)
3. Cosine similarity between cluster and page vectors
4. Primary page = highest similarity
5. Runner-up = second-highest similarity

Cannibalization risk flagged when:
- Match score > 0.12 (some similarity exists)
- Difference between primary and runner-up < 0.03 (too close)

### Recommended Actions

Generated based on:
- **Cannibalization risk**: "Fix cannibalisation"
- **Transactional intent + position > 10**: "Build/upgrade landing page"
- **Position ≤ 10**: "CTR/snippet + internal links"
- **Otherwise**: "Content refresh + topical authority"

## Visualizations

### Bubble Chart

Interactive Plotly chart showing:
- **X-axis**: Average position (reversed, so left = better)
- **Y-axis**: Opportunity clicks
- **Outer bubble (faded)**: Total potential clicks (actual + opportunity)
- **Inner bubble (solid)**: Actual clicks
- **Color**: Priority band (P1=red, P2=orange, P3=blue)

The gap between outer and inner bubbles represents the opportunity.

### Filters (Cascading)

- **Brand Segment** (first): All, Branded, Non-Branded
- **Intent** (second, filtered by brand): Shows only intents available for selected brand

Filters cascade - selecting a brand updates available intents. Opportunity clicks metrics update to reflect brand selection.

### Analysis Tables

**All Keywords Tab:**
- Filterable by Brand → Topic → Intent (cascading filters)
- Shows all individual queries with metrics
- Includes: query, cluster_id, topic_label, segment, intent, brand_label, clicks, impressions, CTR, position, opportunity_clicks, priority_score, priority_band, primary_page
- Sortable by multiple columns

**Cannibalization Risks Tab:**
- Shows competing clusters and the pages they target
- Displays primary vs competing cluster information
- Color-coded risk levels (High/Medium/Low)
- Sorted by score difference (smallest = highest risk)

**Action Priority Tab:**
- Sorted by effort level (low to high), then by opportunity_clicks
- Grouped by effort level with expandable sections
- Shows counts per effort level
- Effort levels: Low (CTR/snippet), Medium (Content refresh), Medium-High (Fix cannibalisation), High (Build/upgrade landing page)

## Technical Details

### Performance Optimizations

- **MiniBatchKMeans** for large segments (>10k queries)
- **Vectorized operations** using pandas/numpy
- **Efficient similarity computation** with scikit-learn

### Data Processing

- **Multi-language column mapping**: Auto-detects and maps CSV columns in multiple languages
- Handles missing CTR values (calculates from clicks/impressions)
- Normalizes queries (lowercase, removes special chars, handles apostrophes)
- Tokenizes URLs for page matching
- Robust error handling for malformed data
- Column mapping validation before processing

### Dependencies

- `streamlit`: Web interface
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: Clustering and vectorization
- `plotly`: Interactive visualizations
- `matplotlib`: Static charts
- `scipy`: Scientific computing (used by sklearn)
- `sentence-transformers` (optional): Semantic clustering support

## Tips for Best Results

1. **GSC Export Limit**: Remember GSC UI limits exports to 1,000 rows. For larger datasets, use the GSC API or export multiple date ranges
2. **Date Range**: Use 3-6 months of data for stable clustering
3. **Brand Terms**: Include all variations (e.g., "acme, acme corp, acme inc")
4. **Data Quality**: Ensure GSC exports include all required columns (in any language)
5. **Column Mapping**: For non-English exports, review auto-detected mappings and adjust if needed
6. **Intent Keywords**: Customize intent keywords to match your industry/niche for better classification
7. **Semantic Clustering**: Consider using semantic clustering for better results (requires sentence-transformers installation)
8. **Singular/Plural Merging**: Enable if you want to merge clusters with singular/plural variations
9. **Review Clusters**: Check topic labels make sense; adjust brand terms if needed
10. **Cannibalization**: Pay attention to flagged clusters; may need canonical tags or content consolidation
11. **Action Priority**: Use the Action Priority table to focus on low-effort, high-impact opportunities first

## Troubleshooting

### "Missing column mappings" error
- **All required fields must be mapped**: Use the column mapping UI to map each required field
- **Auto-detection failed**: If auto-detection doesn't find a column, manually select it from the dropdown
- **Check column names**: Ensure your CSV has columns containing the required data (query/page, clicks, impressions, CTR, position)
- **Supported languages**: Auto-detection works for English, Dutch, French, Spanish, German, and more. For other languages, use manual mapping

### Clustering seems off
- Verify brand terms are correctly configured
- Check that date range has sufficient data (at least 100 queries recommended)
- Review intent classification in the tables tab

### Performance issues
- For very large datasets (>50k queries), processing may take 1-2 minutes
- Consider filtering GSC exports to top queries/pages if needed
- Semantic clustering may be slower on first run (model download), but subsequent runs are faster due to caching

### GSC Export Limitations
- **1,000 row limit**: GSC UI exports are limited to 1,000 rows maximum
- **Workaround**: Use GSC API for larger datasets, or export multiple date ranges and combine
- **Impact**: Smaller datasets may have less stable clustering - consider using semantic clustering for better results

## License

Free tool - use as needed. If you find it useful, connect with the creator on LinkedIn!

## Credits

Built with ❤️ using Streamlit, scikit-learn, and Plotly.
