# Reconstruit les CSV à partir des .partXX (Windows PowerShell)
Get-Content -Path dataset\pre_processed_ECB.csv.part?? -Raw -Encoding Byte |
  Set-Content -Path dataset\pre_processed_ECB.csv -Encoding Byte

Get-Content -Path dataset\pre_processed_ECB_clean_simple.csv.part?? -Raw -Encoding Byte |
  Set-Content -Path dataset\pre_processed_ECB_clean_simple.csv -Encoding Byte
