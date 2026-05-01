"""Quick script to check if the Access AST300 L1 RD Service is running."""
import requests

print("=" * 50)
print("  SCANNER RD SERVICE CHECK")
print("  Device: Access AST300 L1 (FAP20 Thermal)")
print("=" * 50)
print()

found = False
for port in range(11100, 11121):
    for method in ['POST', 'GET']:
        try:
            url = f"http://127.0.0.1:{port}/rd/info"
            if method == 'POST':
                resp = requests.post(url, data='', headers={'Content-Type': 'text/xml'}, timeout=1)
            else:
                resp = requests.get(url, timeout=1)
            
            if resp.status_code == 200:
                print(f"  [OK] Port {port} ({method}): RD Service FOUND!")
                print(f"  Response: {resp.text[:300]}")
                found = True
                break
            elif resp.status_code in [405, 400, 500]:
                # Service is alive but this method/endpoint isn't right
                print(f"  [OK] Port {port}: RD Service DETECTED (HTTP {resp.status_code})")
                found = True
                break
        except requests.ConnectionError:
            pass
        except requests.Timeout:
            pass
    if found:
        break

if not found:
    # Try HTTPS
    for port in range(11100, 11121):
        for method in ['POST', 'GET']:
            try:
                url = f"https://127.0.0.1:{port}/rd/info"
                if method == 'POST':
                    resp = requests.post(url, data='', headers={'Content-Type': 'text/xml'}, timeout=1, verify=False)
                else:
                    resp = requests.get(url, timeout=1, verify=False)
                
                if resp.status_code in [200, 405, 400, 500]:
                    print(f"  [OK] Port {port} (HTTPS/{method}): RD Service FOUND!")
                    if resp.status_code == 200:
                        print(f"  Response: {resp.text[:300]}")
                    found = True
                    break
            except Exception:
                pass
        if found:
            break

if not found:
    print("  [X] RD Service NOT FOUND on ports 11100-11120")
    print()
    print("  To fix this:")
    print("  1. Plug in your Access AST300 L1 scanner via USB")
    print("  2. Install the ACPL L1 RD Service from Access Computech website")
    print("  3. Open Windows Services (Win+R -> services.msc)")
    print("     and start 'ACPL L1 RD Service'")

print()
print("=" * 50)
