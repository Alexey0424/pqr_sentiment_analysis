# PyInstaller specification for the PQR Sentiment Studio desktop build.
# Build with: pyinstaller packaging/pqr_studio.spec (see scripts/build_exe.ps1)

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("customtkinter")
datas += [
    ("../data/pqr_records.csv", "data"),
    ("../data/pqr_scored.csv", "data"),
]

a = Analysis(
    ["launcher.py"],
    pathex=[".."],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=["pytest", "nbformat", "nbclient", "IPython", "jupyter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="PQR_Sentiment_Studio",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="PQR_Sentiment_Studio",
)
