"""Scan I2C bus 1 and report which addresses answer.

Usage:  python tools/bringup/i2c_scan.py [--bus 1] [--expect 0x40]
Exit code 0 if every --expect address answered, 1 otherwise.
"""
import argparse
import sys
from smbus2 import SMBus 

def scan(bus_number: int) -> list[int]:
    """Return the addresses (0x03..0x77) that acknowledge on this bus."""
    with SMBus(bus_number) as bus:
        found_addresses = []
        for address in range(0x03, 0x78):  # Valid 7-bit I2C addresses
            try:
                bus.read_byte(address)  # Try a harmless read
                found_addresses.append(address)
            except OSError:
                print(f"Address {address:#04x} did not respond. Nobody home.")
                continue  # No device at this address
    return found_addresses

def main() -> int:

    argparser = argparse.ArgumentParser(description="Scan I2C bus for devices.")
    argparser.add_argument('--bus', type=int, default=1, help='I2C bus number to scan (default: 1)')
    argparser.add_argument('--expect', type=str, nargs='*', help='Expected I2C addresses in hex (e.g., 0x40 0x50)')
    args = argparser.parse_args()
    scan_results = scan(args.bus)

    for address in scan_results:
        print(f"Address {address:#04x} responded.")

    if args.expect:
        expected_addresses = [int(addr, 16) for addr in args.expect]
        missing_addresses = set(expected_addresses) - set(scan_results)
        for address in missing_addresses:
            print(f"Expected address {address:#04x} did not respond.")
        return 1 if missing_addresses else 0
    return 0

if __name__ == "__main__":
    sys.exit(main())