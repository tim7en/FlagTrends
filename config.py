"""
Configuration: analysis parameters and Libertex -> yfinance symbol mapping.
"""

# === Analysis Parameters ===
LOOKBACK_DAYS = 365      # Days of historical data to fetch
SHORT_MA_PERIOD = 20     # Short EMA period
MEDIUM_MA_PERIOD = 50    # Medium SMA period
LONG_MA_PERIOD = 200     # Long SMA period
SIGNAL_WINDOW = 5        # Days to look back for a recent break signal
DONCHIAN_PERIOD = 20     # Period for Donchian channel breakout check
FETCH_WORKERS = 10       # Parallel download threads

# === yfinance Symbol Mapping ===
# Maps Libertex symbols -> yfinance tickers
SYMBOL_MAP: dict[str, str] = {
    # ── Commodities ──────────────────────────────────────────────────────────
    "BRN":          "BZ=F",      # Brent Crude Oil
    "CL":           "CL=F",      # Light Sweet Crude Oil (WTI front-month)
    "COCOA":        "CC=F",      # Cocoa
    "COFFEE":       "KC=F",      # Coffee C
    "CORN":         "ZC=F",      # Corn
    "HG":           "HG=F",      # Copper
    "HO":           "HO=F",      # Heating Oil
    "NG":           "NG=F",      # Henry Hub Natural Gas
    "PA":           "PA=F",      # Palladium
    "PL":           "PL=F",      # Platinum
    "SOYBEAN":      "ZS=F",      # Soybean
    "SUGAR":        "SB=F",      # Sugar #11
    "WHEAT":        "ZW=F",      # Wheat
    "WT":           "CL=F",      # WTI Crude (same futures contract as CL)
    "XAGUSD":       "SI=F",      # Silver
    "XAUUSD":       "GC=F",      # Gold

    # ── US / ADR stocks ───────────────────────────────────────────────────────
    "AA":           "AA",
    "AAPL":         "AAPL",
    "ABEV":         "ABEV",
    "ACB":          "ACB",
    "ADBE":         "ADBE",
    "AMC":          "AMC",
    "AMD":          "AMD",
    "AMZN":         "AMZN",
    "ATVI":         "ATVI",
    "AXP":          "AXP",
    "BA":           "BA",
    "BABA":         "BABA",
    "BAC":          "BAC",
    "BIDU":         "BIDU",
    "BNPQY":        "BNPQY",
    "BSAC":         "BSAC",
    "C":            "C",
    "CAT":          "CAT",
    "CCL":          "CCL",
    "COIN":         "COIN",
    "CRM":          "CRM",
    "CRON":         "CRON",
    "CSCO":         "CSCO",
    "DIS":          "DIS",
    "Dropbox":      "DBX",
    "EBAY":         "EBAY",
    "EL":           "EL",
    "ENIC":         "ENIC",
    "ETH":          "ETH",       # Ethan Allen (NOT crypto)
    "F":            "F",
    "FB":           "META",      # Facebook → Meta
    "FCX":          "FCX",
    "GBTC":         "GBTC",
    "GE":           "GE",
    "GILD":         "GILD",
    "GME":          "GME",
    "GOOGL":        "GOOGL",
    "GS":           "GS",
    "HD":           "HD",
    "HOG":          "HOG",
    "HOOD":         "HOOD",
    "HPQ":          "HPQ",
    "IBM":          "IBM",
    "INTC":         "INTC",
    "JNJ":          "JNJ",
    "JPM":          "JPM",
    "JWN":          "JWN",
    "KO":           "KO",
    "KORS":         "CPRI",      # Michael Kors → Capri Holdings
    "LUV":          "LUV",
    "LYFT":         "LYFT",
    "MA":           "MA",
    "MARA":         "MARA",
    "MCD":          "MCD",
    "MRNA":         "MRNA",
    "MSFT":         "MSFT",
    "MU":           "MU",
    "NCLH":         "NCLH",
    "NINTENDO_US":  "NTDOY",
    "NKE":          "NKE",
    "NVDA":         "NVDA",
    "Netflix":      "NFLX",
    "ORCL":         "ORCL",
    "PBR":          "PBR",
    "PFE":          "PFE",
    "PG":           "PG",
    "PINS":         "PINS",
    "PM":           "PM",
    "PVH":          "PVH",
    "RACE":         "RACE",
    "RCL":          "RCL",
    "RL":           "RL",
    "RNLSY":        "RNLSY",
    "RYAAY":        "RYAAY",
    "SAP":          "SAP",
    "SAVE":         "SAVE",
    "SBUX":         "SBUX",
    "SIRI":         "SIRI",
    "SNAP":         "SNAP",
    "SPCE":         "SPCE",
    "SQM":          "SQM",
    "SVMK":         "MNTV",      # SurveyMonkey → Momentive
    "Spotify":      "SPOT",
    "T":            "T",
    "TLRY":         "TLRY",
    "TM":           "TM",
    "TRIP":         "TRIP",
    "TRV":          "TRV",
    "TSLA":         "TSLA",
    "UBER":         "UBER",
    "UNH":          "UNH",
    "V":            "V",
    "VALE":         "VALE",
    "VFC":          "VFC",
    "VZ":           "VZ",
    "WEED":         "CGC",       # Canopy Growth (US listing)
    "WFC":          "WFC",
    "WSM":          "WSM",
    "WYNN":         "WYNN",
    "XOM":          "XOM",

    # ── European stocks (exchange suffix) ─────────────────────────────────────
    "ADS":          "ADS.DE",    # Adidas
    "AIR":          "AIR.PA",    # Airbus
    "AIRF":         "AF.PA",     # Air France-KLM
    "BAS":          "BAS.DE",    # BASF
    "BAYN":         "BAYN.DE",   # Bayer
    "BMW":          "BMW.DE",
    "DAI":          "MBG.DE",    # Mercedes-Benz (formerly Daimler)
    "DBK":          "DBK.DE",    # Deutsche Bank
    "DG":           "DG.PA",     # Vinci SA
    "ENEL":         "ENEL.MI",   # Enel SpA
    "ENI":          "ENI.MI",    # Eni SpA
    "FP":           "TTE.PA",    # TotalEnergies
    "ITX":          "ITX.MC",    # Inditex
    "REP":          "REP.MC",    # Repsol
    "SIE":          "SIE.DE",    # Siemens
    "TUI":          "TUI1.DE",
    "VOW":          "VOW3.DE",   # Volkswagen ordinary shares

    # ── Asian stocks ──────────────────────────────────────────────────────────
    "IDCB":         "1398.HK",   # ICBC
    "LNVG":         "0992.HK",   # Lenovo
    "NINTENDO_JP":  "7974.T",    # Nintendo (Tokyo)
    "TCTZ":         "0700.HK",   # Tencent
}
