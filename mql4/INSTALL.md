# TradeForge MQL4 Installation Guide

## Prerequisites

- MetaTrader 4 (MT4) installed on your machine
- The TradeForge MQL4 files from this repository

---

## Step 1: Locate Your MT4 Data Folder

1. Open MetaTrader 4
2. In the top menu, click **File > Open Data Folder**
3. This opens the MT4 data directory (e.g. `C:\Users\<YourName>\AppData\Roaming\MetaQuotes\Terminal\<ID>\MQL4\`)

---

## Step 2: Copy the Files

Copy the contents of this `mql4/` folder into your MT4 `MQL4/` data folder, merging with the existing structure:

| Copy from (this repo)        | Paste into (MT4 data folder)  |
|------------------------------|-------------------------------|
| `mql4/Include/`              | `MQL4/Include/`               |
| `mql4/Libraries/`            | `MQL4/Libraries/`             |
| `mql4/Experts/`              | `MQL4/Experts/`               |

> **Note:** When prompted, choose **Merge** — do not replace the entire folder, only add the new files.

---

## Step 3: Enable Expert Advisors in MT4 Settings

This step is required for the Expert Advisor (EA) to run and communicate properly.

1. In MT4, go to **Tools > Options** (or press `Ctrl+O`)
2. Click the **Expert Advisors** tab
3. Enable the following settings:
   - **Allow Automated Trading** — lets the EA place and manage trades
   - **Allow DLL imports** — required for `libzmq.dll` and `libsodium.dll` to load correctly
4. Click **OK** to save

---

## Step 4: Enable the Automated Trading Button

On the MT4 toolbar, make sure the **Automated Trading** button is active (it should appear green/enabled). Click it to toggle it on if it is not already.

---

## Step 5: Attach the Expert Advisor

1. In the MT4 **Navigator** panel (left sidebar), expand **Expert Advisors**
2. Drag your EA onto a chart, or double-click it to open the settings dialog
3. In the EA settings dialog, go to the **Common** tab and confirm:
   - **Allow live trading** is checked
   - **Allow DLL imports** is checked
4. Click **OK**

---

## Included Libraries

| File                        | Purpose                                      |
|-----------------------------|----------------------------------------------|
| `Libraries/libzmq.dll`      | ZeroMQ messaging library for live data/comms |
| `Libraries/libsodium.dll`   | Cryptographic library used by ZeroMQ         |
| `Include/Zmq/`              | MQL4 ZeroMQ wrapper headers                  |
| `Include/JAson.mqh`         | JSON parsing library                         |
| `Include/Mql/`              | MQL4 utility helpers                         |

---

## Troubleshooting

- **EA shows a grey/red face icon on the chart** — Automated Trading is disabled. Enable it via the toolbar button or the Expert Advisors settings (Step 3 & 4).
- **"DLL not allowed" error in the Experts log** — Allow DLL imports is not enabled. Revisit Step 3.
- **EA loads but does nothing** — Check the **Experts** and **Journal** tabs at the bottom of MT4 for error messages.
- **Cannot find the EA in Navigator** — Make sure you copied the `.ex4` or `.mq4` file to the correct `MQL4/Experts/` folder and refreshed Navigator (right-click > Refresh).
