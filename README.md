

## Google Drive download troubleshooting

CyberlawGPT automatically tries to download the configured Google Drive PDF at startup. The Drive file must be shared as **General access → Anyone with the link → Viewer** for Streamlit Cloud to download it without authentication.

If your Google Workspace/organization does not allow public Drive files, use the **Upload cyber-law PDF (fallback)** control in the Streamlit sidebar. This keeps the project to the same three files and does not require changing the Python code.

The app validates that the downloaded file starts with the PDF signature (`%PDF-`) so an HTML Google Drive permission page is not accidentally indexed as a PDF.
