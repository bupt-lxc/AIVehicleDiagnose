"""测试 XML 解析器解析 test1.xml"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.diagnostic.tools.xml_report_parser import parse_diagnostic_report, convert_to_structured_data

xml_path = "testing/test1.xml"
print(f"解析文件: {xml_path}")
print(f"文件大小: {os.path.getsize(xml_path):,} bytes")

report = parse_diagnostic_report(xml_path)
if report is None:
    print("ERROR: 解析失败!")
    sys.exit(1)

print("\n=== 基本信息 ===")
bi = report.basic_info
print(f"  VIN: {bi.vin}")
print(f"  User: {bi.user}")
print(f"  Project: {bi.readout_project}")
print(f"  Date: {bi.readout_date} {bi.readout_time}")
print(f"  Mileage: {bi.mileage} {bi.mileage_unit}")
print(f"  Version: {bi.readout_version}")
print(f"  Filename: {bi.filename}")
print(f"  总DTC数量: {bi.total_dtc_count}")

print(f"\n=== ECU信息: {len(report.ecu_info)} 个ECU ===")
active_dtcs = 0
passive_dtcs = 0
for addr, ecu in sorted(report.ecu_info.items()):
    status = "Online" if ecu.no_contact != "1" else "NO CONTACT"
    print(f"  ECU {addr}: {ecu.ecu_description[:60]}")
    print(f"    SW: {ecu.sw_version} | HW: {ecu.hw_version}")
    print(f"    VKMS: {ecu.is_vkms} | SOK: {ecu.is_sok} | NoBasicSetting: {ecu.is_no_basic_setting}")
    print(f"    DTCs: {len(ecu.dtc_info)} (Aktiv: {ecu.aktiv_count}, Passiv: {ecu.passiv_count})")
    active_dtcs += ecu.aktiv_count
    passive_dtcs += ecu.passiv_count
    for dtc in ecu.dtc_info[:3]:
        print(f"      DTC #{dtc.dtc_number_dec}: {dtc.dtc_odx_text[:60]} [{dtc.fault_status}] [{dtc.fault_event}]")
    if len(ecu.dtc_info) > 3:
        print(f"      ... 还有 {len(ecu.dtc_info) - 3} 个DTC")
    if len(ecu.sub_ecus) > 0:
        print(f"    Sub ECUs: {len(ecu.sub_ecus)}")
    print()

print(f"总计: Active DTCs={active_dtcs}, Passive DTCs={passive_dtcs}")

print("\n=== 转换为 StructuredData 格式 ===")
structured = convert_to_structured_data(report)
print(f"  DTC Codes: {len(structured['dtc_codes'])}")
print(f"  ECU States: {len(structured['ecu_states'])}")
print(f"  Freeze Frames: {len(structured['freeze_frames'])}")

# 按 domain 统计 ECU
domain_count = {}
for ecu in structured['ecu_states']:
    d = ecu.get('domain', 'Unknown')
    domain_count[d] = domain_count.get(d, 0) + 1
print(f"  ECU Domain分布: {domain_count}")

# 统计 Active/Stored DTC
active_count = sum(1 for d in structured['dtc_codes'] if d.get('is_active'))
stored_count = len(structured['dtc_codes']) - active_count
print(f"  Active DTCs: {active_count}, Stored: {stored_count}")

# 列出前20个DTC
print("\n前20个DTC:")
for i, dtc in enumerate(structured['dtc_codes'][:20]):
    print(f"  [{i+1}] {dtc['code']} | {dtc['status']} | ECU: {dtc['ecu_id']} | {dtc.get('description', '')[:50]}")

# 保存转换结果
output_path = "logs/test1_parsed.json"
os.makedirs("logs", exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(structured, f, ensure_ascii=False, indent=2, default=str)
print(f"\n解析结果已保存到: {output_path}")
print(f"\nXML解析器测试: PASS")