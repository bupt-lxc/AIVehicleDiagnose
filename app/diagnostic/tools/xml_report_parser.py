"""XML电检报告解析器

用于解析德国汽车诊断系统（大众/奥迪）生成的XML格式诊断报告。
"""

import os
import xml.etree.ElementTree as ET
from typing import Optional
from dataclasses import dataclass, field
from loguru import logger

logger = logger.bind(component="XMLReportParser")


@dataclass
class DTCData:
    """DTC故障码数据"""
    dtc_number_dec: str = "-8"
    fehler_typ: str = "-8"
    dtc_odx_text: str = "-8"
    dtc_vas_text: str = "-8"
    odx_error: str = "-8"
    dtc_priority: str = "-8"
    dtc_kind_color: str = "-8"
    fault_event: str = "Unknown"
    fault_status: str = "Unknown"


@dataclass
class SubECUData:
    """子ECU数据"""
    sub_ecu_name: str = "-8"
    sub_ecu_hw_version: str = "-8"
    sub_ecu_hw_partnumber: str = "-8"
    sub_ecu_sw_version: str = "-8"
    sub_ecu_sw_partnumber: str = "-8"
    sub_ecu_codierung: str = "-8"
    sub_ecu_schluesseltyp: str = "-8"
    sub_ecu_vkms_status: str = "-8"
    sub_ecu_sfd_role: str = "-8"
    sub_ecu_sfd_duration: str = "-8"
    sub_ecu_fazit_string: str = "-8"
    sub_ecu_zdc_name: str = "-8"
    sub_ecu_zdc_version: str = "-8"


@dataclass
class ECUInfo:
    """ECU信息"""
    hw_address: str = "-8"
    ecu_description: str = "-8"
    sw_partnumber: str = "-8"
    sw_version: str = "-8"
    bootloader_version: str = "-8"
    hw_partnumber: str = "-8"
    hw_version: str = "-8"
    codierung: str = "-8"
    schluesseltyp: str = "-8"
    vkms_status: str = "-8"
    sfd_role: str = "-8"
    sfd_duration: str = "-8"
    fazit_string: str = "-8"
    zdc_name: str = "-8"
    zdc_version: str = "-8"
    no_contact: str = "-8"
    sub_ecus: dict = field(default_factory=dict)
    dtc_info: list = field(default_factory=list)
    ereignisse_count: int = 0
    informationen_count: int = 0
    aktiv_count: int = 0
    passiv_count: int = 0
    is_vkms: bool = False
    is_sok: bool = False
    is_no_basic_setting: bool = False


@dataclass
class BasicInfo:
    """基本信息"""
    user: str = "-8"
    readout_version: str = "-8"
    readout_date: str = "-8"
    readout_time: str = "-8"
    readout_project: str = "-8"
    vin: str = "-8"
    mileage: str = "-8"
    mileage_unit: str = "-8"
    total_dtc_count: int = 0
    filename: str = "-8"


@dataclass
class ParsedReport:
    """解析后的报告数据"""
    basic_info: BasicInfo = field(default_factory=BasicInfo)
    ecu_info: dict = field(default_factory=dict)


def classify_fault(dtc_priority: str, dtc_kind_color: str) -> tuple:
    """
    根据DTC优先级和类型颜色分类故障
    
    Args:
        dtc_priority: DTC优先级
        dtc_kind_color: DTC类型颜色
        
    Returns:
        (fault_event, fault_status) 元组
    """
    try:
        priority = int(dtc_priority) if dtc_priority and dtc_priority != '-8' else 0
        kind_color = int(dtc_kind_color) if dtc_kind_color and dtc_kind_color != '-8' else 0
    except (ValueError, TypeError):
        return ('Unknown', 'Unknown')
    
    if kind_color == 11:
        fault_status = 'Aktiv'
    elif kind_color == 12:
        fault_status = 'Passiv'
    else:
        fault_status = 'Unknown'
    
    if priority in [1, 2, 3]:
        fault_event = 'Ereignisse'
    elif priority in [4, 5, 6]:
        fault_event = 'Informationen'
    else:
        fault_event = 'Unknown'
    
    return (fault_event, fault_status)


def _safe_get_text(element: ET.Element | None, tag: str, default: str = "-8") -> str:
    """安全获取XML元素文本"""
    if element is None:
        return default
    child = element.find(tag)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _safe_get_nested_text(element: ET.Element | None, path: str, default: str = "-8") -> str:
    """安全获取嵌套XML元素文本"""
    if element is None:
        return default
    child = element.find(path)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def parse_diagnostic_report(xml_path: str) -> Optional[ParsedReport]:
    """
    解析电检报告XML文件，返回结构化数据
    
    Args:
        xml_path: XML文件路径
        
    Returns:
        ParsedReport对象，解析失败返回None
    """
    result = ParsedReport()
    
    try:
        with open(xml_path, 'rb') as f:
            tree = ET.parse(f)
        root = tree.getroot()
        
        result.basic_info = BasicInfo(
            user=_safe_get_text(root, 'User'),
            readout_version=_safe_get_text(root, 'Report-Version'),
            readout_date=_safe_get_text(root, 'Datum'),
            readout_time=_safe_get_text(root, 'Zeit'),
            readout_project=_safe_get_text(root, 'UserProjekt'),
            vin=_safe_get_text(root, 'Fahrgestellnummer'),
            mileage=_safe_get_text(root, 'WegStrecke'),
            mileage_unit=_safe_get_text(root, 'EinheitWegStrecke'),
            total_dtc_count=0,
            filename=os.path.basename(xml_path)
        )
        
        ecu_blocks = root.findall('Diagnosebloecke/Diagnoseblock')
        if ecu_blocks:
            for ecu in ecu_blocks:
                hw_address = _safe_get_text(ecu, 'Adresse')
                
                ecu_info = ECUInfo(
                    hw_address=hw_address,
                    ecu_description=_safe_get_text(ecu, 'Systembezeichnung'),
                    sw_partnumber=_safe_get_text(ecu, 'SWTeilenummer'),
                    sw_version=_safe_get_text(ecu, 'SWVersion'),
                    bootloader_version=_safe_get_text(ecu, 'BootLoaderVersion'),
                    hw_partnumber=_safe_get_text(ecu, 'HWTeilenummer'),
                    hw_version=_safe_get_text(ecu, 'HWVersion'),
                    codierung=_safe_get_text(ecu, 'Codierung'),
                    schluesseltyp=_safe_get_text(ecu, 'Schluesseltyp'),
                    vkms_status=_safe_get_text(ecu, 'VkmsStatus'),
                    sfd_role=_safe_get_nested_text(ecu, 'Sfd/SfdRole'),
                    sfd_duration=_safe_get_nested_text(ecu, 'Sfd/SfdDuration'),
                    fazit_string=_safe_get_text(ecu, 'FAZITString'),
                    zdc_name=_safe_get_text(ecu, 'ZdcName'),
                    zdc_version=_safe_get_text(ecu, 'ZdcVersion'),
                    no_contact=_safe_get_text(ecu, 'NOCONTACT')
                )
                
                sub_teilnehmer = ecu.find('SubTeilnehmer')
                if sub_teilnehmer is not None:
                    sub_ecus = sub_teilnehmer.findall('Sub')
                    for sub_ecu in sub_ecus:
                        sub_hw_address = _safe_get_text(sub_ecu, 'Adresse')
                        ecu_info.sub_ecus[sub_hw_address] = SubECUData(
                            sub_ecu_name=_safe_get_text(sub_ecu, 'SubtName'),
                            sub_ecu_hw_version=_safe_get_text(sub_ecu, 'HWVersion'),
                            sub_ecu_hw_partnumber=_safe_get_text(sub_ecu, 'HWTeilenummer'),
                            sub_ecu_sw_version=_safe_get_text(sub_ecu, 'SWVersion'),
                            sub_ecu_sw_partnumber=_safe_get_text(sub_ecu, 'SWTeilenummer'),
                            sub_ecu_codierung=_safe_get_text(sub_ecu, 'Codierung'),
                            sub_ecu_schluesseltyp=_safe_get_text(sub_ecu, 'Schluesseltyp'),
                            sub_ecu_vkms_status=_safe_get_text(sub_ecu, 'VkmsStatus'),
                            sub_ecu_sfd_role=_safe_get_nested_text(sub_ecu, 'Sfd/SfdRole'),
                            sub_ecu_sfd_duration=_safe_get_nested_text(sub_ecu, 'Sfd/SfdDuration'),
                            sub_ecu_fazit_string=_safe_get_text(sub_ecu, 'FAZITString'),
                            sub_ecu_zdc_name=_safe_get_text(sub_ecu, 'ZdcName'),
                            sub_ecu_zdc_version=_safe_get_text(sub_ecu, 'ZdcVersion')
                        )
                
                dtcs = ecu.findall('Fehlerspeicher/Block/Fehler/Fehler')
                if dtcs:
                    for dtc in dtcs:
                        dtc_data = DTCData(
                            dtc_number_dec=_safe_get_text(dtc, 'DTCNummer_Dec'),
                            fehler_typ=_safe_get_text(dtc, 'FehlerTyp'),
                            dtc_odx_text=_safe_get_text(dtc, 'DTCODXTEXT'),
                            dtc_vas_text=_safe_get_text(dtc, 'DTCVASTEXT'),
                            odx_error=_safe_get_text(dtc, 'ODXError'),
                            dtc_priority=_safe_get_text(dtc, 'DtcPriority'),
                            dtc_kind_color=_safe_get_text(dtc, 'DTCKIND_COLOR')
                        )
                        
                        fault_event, fault_status = classify_fault(
                            dtc_data.dtc_priority, 
                            dtc_data.dtc_kind_color
                        )
                        dtc_data.fault_event = fault_event
                        dtc_data.fault_status = fault_status
                        
                        ecu_info.dtc_info.append(dtc_data)
                        
                        if "VKMS" in dtc_data.dtc_odx_text or "VKMS" in dtc_data.dtc_vas_text:
                            ecu_info.is_vkms = True
                        elif "SOK" in dtc_data.dtc_odx_text or "SOK" in dtc_data.dtc_vas_text:
                            ecu_info.is_sok = True
                        elif "no basic setting" in dtc_data.dtc_odx_text or "no basic setting" in dtc_data.dtc_vas_text:
                            ecu_info.is_no_basic_setting = True
                        
                        if fault_status == 'Aktiv':
                            result.basic_info.total_dtc_count += 1
                            ecu_info.aktiv_count += 1
                            if fault_event == 'Ereignisse':
                                ecu_info.ereignisse_count += 1
                            elif fault_event == 'Informationen':
                                ecu_info.informationen_count += 1
                        elif fault_status == 'Passiv':
                            result.basic_info.total_dtc_count += 1
                            ecu_info.passiv_count += 1
                
                result.ecu_info[hw_address] = ecu_info
        
        logger.info("[XMLReportParser] 解析成功: {} 个ECU, {} 个DTC", 
                   len(result.ecu_info), result.basic_info.total_dtc_count)
        
        return result
        
    except Exception as e:
        logger.error("[XMLReportParser] 解析电检报告出错: {}", e)
        return None


def convert_to_structured_data(report: ParsedReport) -> dict:
    """
    将ParsedReport转换为ParserAgent兼容的StructuredData格式
    
    Args:
        report: 解析后的报告对象
        
    Returns:
        StructuredData格式的字典
    """
    dtc_codes = []
    ecu_states = []
    
    for hw_address, ecu_info in report.ecu_info.items():
        ecu_states.append({
            "ecu_id": hw_address,
            "domain": _infer_domain(ecu_info.ecu_description),
            "status": "Online" if ecu_info.no_contact != "1" else "Offline",
            "communication_status": "Normal" if ecu_info.no_contact != "1" else "No Contact",
            "power_segment": None
        })
        
        for dtc in ecu_info.dtc_info:
            status = "Active" if dtc.fault_status == "Aktiv" else "Stored"
            event_type = "Event" if dtc.fault_event == "Ereignisse" else "Info"
            is_active = dtc.fault_status == "Aktiv"
            
            dtc_codes.append({
                "code": _convert_dtc_code(dtc.dtc_number_dec),
                "status": status,
                "description": dtc.dtc_odx_text or dtc.dtc_vas_text,
                "ecu_id": hw_address,
                "event_type": event_type,
                "is_active": is_active,
                "raw_data": {
                    "dtc_number_dec": dtc.dtc_number_dec,
                    "fehler_typ": dtc.fehler_typ,
                    "dtc_priority": dtc.dtc_priority,
                    "fault_event": dtc.fault_event,
                    "fault_status": dtc.fault_status
                }
            })
    
    return {
        "dtc_codes": dtc_codes,
        "ecu_states": ecu_states,
        "freeze_frames": [],
        "raw_text_segments": [],
        "basic_info": {
            "vin": report.basic_info.vin,
            "mileage": report.basic_info.mileage,
            "mileage_unit": report.basic_info.mileage_unit,
            "readout_date": report.basic_info.readout_date,
            "readout_time": report.basic_info.readout_time,
            "user": report.basic_info.user,
            "project": report.basic_info.readout_project
        }
    }


def _infer_domain(ecu_description: str) -> str:
    """根据ECU描述推断域"""
    desc_lower = ecu_description.lower()
    
    if any(kw in desc_lower for kw in ["engine", "transmission", "motor", "getriebe", "drive", "battery", "bms", "hybrid", "electric"]):
        return "Powertrain"
    elif any(kw in desc_lower for kw in ["brake", "abs", "esp", "steering", "lenkung", "bremse", "chassis", "suspension"]):
        return "Chassis"
    elif any(kw in desc_lower for kw in ["door", "seat", "light", "window", "mirror", "body", "comfort", "tuer", "licht", "sitze"]):
        return "Body"
    elif any(kw in desc_lower for kw in ["infotainment", "navigation", "radio", "display", "audio", "media"]):
        return "Infotainment"
    elif any(kw in desc_lower for kw in ["acc", "radar", "camera", "adas", "assist", "autonomous", "lidar"]):
        return "ADAS"
    else:
        return "Unknown"


def _convert_dtc_code(dtc_number_dec: str) -> str:
    """
    将十进制DTC码转换为标准格式
    
    大众/奥迪的DTC码是十进制格式，需要转换为标准P/C/B/U格式
    """
    try:
        dec_value = int(dtc_number_dec)
        
        hex_value = hex(dec_value)[2:].upper().zfill(6)
        
        if len(hex_value) >= 4:
            first_nibble = hex_value[0]
            second_char = hex_value[1]
            remaining = hex_value[2:6]
            
            prefix_map = {
                '0': 'P',
                '1': 'C',
                '2': 'B',
                '3': 'U'
            }
            prefix = prefix_map.get(first_nibble, 'P')
            
            return f"{prefix}{second_char}{remaining}"
        
        return f"P{dtc_number_dec}"
        
    except (ValueError, TypeError):
        return f"P{dtc_number_dec}"


def is_xml_file(input_path: str) -> bool:
    """检查输入是否为有效的XML文件路径"""
    if not input_path:
        return False
    
    if not os.path.isfile(input_path):
        return False
    
    if not input_path.lower().endswith('.xml'):
        return False
    
    try:
        with open(input_path, 'rb') as f:
            header = f.read(100)
            if b'<?xml' in header or b'<Protokoll' in header:
                return True
    except Exception:
        pass
    
    return False
