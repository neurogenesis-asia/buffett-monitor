import sys
sys.path.insert(0, '.')
from buffett.fetchers import fetch_malaysiastock_price
print("Testing KLSE scraper...")
test_codes = ["1155", "1295", "5347", "5681", "1023"]
for code in test_codes:
    price = fetch_malaysiastock_price(code)
    if price:
        print(f"{code}: RM {price:.3f}")
    else:
        print(f"{code}: Failed")
