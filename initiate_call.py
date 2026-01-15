#!/usr/bin/env python3
"""
Script to initiate outbound calls for Medicaid renewal
"""

import requests
import argparse
import json
from dotenv import load_dotenv
import os

load_dotenv()

PUBLIC_URL = os.getenv('PUBLIC_URL')

def make_call(to_number, member_id='12345'):
    url = f"{PUBLIC_URL}/make-call"
    
    payload = {
        "to_number": to_number,
        "member_id": member_id
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        print("✅ Call initiated successfully!")
        print(f"   Call SID: {result.get('call_sid')}")
        print(f"   Status: {result.get('status')}")
        print(f"\n📞 Calling {to_number}...")
        print("\nMonitor the call in your Twilio console:")
        print(f"   https://console.twilio.com/us1/monitor/logs/calls/{result.get('call_sid')}")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error making call: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return None

def batch_call(phone_numbers_file):
    """
    Make multiple calls from a file
    
    Args:
        phone_numbers_file: Path to file with phone numbers (one per line)
    """
    try:
        with open(phone_numbers_file, 'r') as f:
            phone_numbers = [line.strip() for line in f if line.strip()]
        
        print(f"📋 Found {len(phone_numbers)} phone numbers")
        print("=" * 60)
        
        results = []
        for i, number in enumerate(phone_numbers, 1):
            print(f"\n[{i}/{len(phone_numbers)}] Calling {number}...")
            result = make_call(number)
            results.append({
                'phone': number,
                'success': result is not None,
                'call_sid': result.get('call_sid') if result else None
            })
            
            # Wait a bit between calls
            if i < len(phone_numbers):
                import time
                time.sleep(2)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 BATCH CALL SUMMARY")
        print("=" * 60)
        
        successful = sum(1 for r in results if r['success'])
        print(f"Total calls: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        
        # Save results
        with open('batch_call_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nResults saved to: batch_call_results.json")
    
    except FileNotFoundError:
        print(f"❌ File not found: {phone_numbers_file}")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Initiate Medicaid renewal calls',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Call a single number
  python initiate_call.py +15555551234
  
  # Call with specific member ID
  python initiate_call.py +15555551234 --member-id 67890
  
  # Batch call from file
  python initiate_call.py --batch phone_numbers.txt
        """
    )
    
    parser.add_argument(
        'phone_number',
        nargs='?',
        help='Phone number to call (E.164 format: +15555551234)'
    )
    
    parser.add_argument(
        '--member-id',
        default='12345',
        help='Member ID (default: 12345)'
    )
    
    parser.add_argument(
        '--batch',
        help='File containing phone numbers (one per line)'
    )
    
    args = parser.parse_args()
    
    if args.batch:
        batch_call(args.batch)
    elif args.phone_number:
        make_call(args.phone_number, args.member_id)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()