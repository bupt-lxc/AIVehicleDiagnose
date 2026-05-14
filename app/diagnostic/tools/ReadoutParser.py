import os
import xml.etree.ElementTree as ET


def classify_fault(dtc_priority: str, dtc_kind_color: str) -> tuple:
    """
    根据DTC优先级和类型颜色分类故障
    :param dtc_priority: DTC优先级
    :param dtc_kind_color: DTC类型颜色
    :return: (fault_event, fault_status) 元组
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


def parse_diagnostic_report(xml_path: str) -> dict:
    """
    解析电检报告XML文件，返回结构化数据（用于langgraph节点）
    :param xml_path: XML文件路径
    :return: 包含解析结果的字典
    """
    result = {
        'basic_info': {},
        'ecu_info': {}
    }

    try:
        with open(xml_path, 'rb') as f:
            tree = ET.parse(f)
        root = tree.getroot()

        result['basic_info'] = {
            'User': root.find('User').text.strip() if root.find('User') is not None and root.find('User').text else '-8',
            'readout_version': root.find('Report-Version').text.strip() if root.find('Report-Version') is not None and root.find('Report-Version').text else '-8',
            'readout_date': root.find('Datum').text.strip() if root.find('Datum') is not None and root.find('Datum').text else '-8',
            'readout_time': root.find('Zeit').text.strip() if root.find('Zeit') is not None and root.find('Zeit').text else '-8',
            'readout_project': root.find('UserProjekt').text.strip() if root.find('UserProjekt') is not None and root.find('UserProjekt').text else '-8',
            'vin': root.find('Fahrgestellnummer').text.strip() if root.find('Fahrgestellnummer') is not None and root.find('Fahrgestellnummer').text else '-8',
            'mileage': root.find('WegStrecke').text.strip() if root.find('WegStrecke') is not None and root.find('WegStrecke').text else '-8',
            'mileage_unit': root.find('EinheitWegStrecke').text.strip() if root.find('EinheitWegStrecke') is not None and root.find('EinheitWegStrecke').text else '-8',
            'total_dtc_count': 0,
            'filename': os.path.basename(xml_path)
        }

        ecu_blocks = root.findall('Diagnosebloecke/Diagnoseblock')
        if ecu_blocks:
            for ecu in ecu_blocks:
                hw_address = ecu.find('Adresse').text.strip() if ecu.find('Adresse') is not None and ecu.find('Adresse').text else '-8'

                ecu_info = {
                    'ecu_description': ecu.find('Systembezeichnung').text.strip() if ecu.find('Systembezeichnung') is not None and ecu.find('Systembezeichnung').text else '-8',
                    'SW_partnumber': ecu.find('SWTeilenummer').text.strip() if ecu.find('SWTeilenummer') is not None and ecu.find('SWTeilenummer').text else '-8',
                    'SW_version': ecu.find('SWVersion').text.strip() if ecu.find('SWVersion') is not None and ecu.find('SWVersion').text else '-8',
                    'BootLoader_version': ecu.find('BootLoaderVersion').text.strip() if ecu.find('BootLoaderVersion') is not None and ecu.find('BootLoaderVersion').text else '-8',
                    'HW_partnumber': ecu.find('HWTeilenummer').text.strip() if ecu.find('HWTeilenummer') is not None and ecu.find('HWTeilenummer').text else '-8',
                    'HW_version': ecu.find('HWVersion').text.strip() if ecu.find('HWVersion') is not None and ecu.find('HWVersion').text else '-8',
                    'Codierung': ecu.find('Codierung').text.strip() if ecu.find('Codierung') is not None and ecu.find('Codierung').text else '-8',
                    'Schluesseltyp': ecu.find('Schluesseltyp').text.strip() if ecu.find('Schluesseltyp') is not None and ecu.find('Schluesseltyp').text else '-8',
                    'VkmsStatus': ecu.find('VkmsStatus').text.strip() if ecu.find('VkmsStatus') is not None and ecu.find('VkmsStatus').text else '-8',
                    'SfdRole': ecu.find('Sfd/SfdRole').text.strip() if ecu.find('Sfd/SfdRole') is not None and ecu.find('Sfd/SfdRole').text else '-8',
                    'SfdDuration': ecu.find('Sfd/SfdDuration').text.strip() if ecu.find('Sfd/SfdDuration') is not None and ecu.find('Sfd/SfdDuration').text else '-8',
                    'FAZIT_string': ecu.find('FAZITString').text.strip() if ecu.find('FAZITString') is not None and ecu.find('FAZITString').text else '-8',
                    'ZdcName': ecu.find('ZdcName').text.strip() if ecu.find('ZdcName') is not None and ecu.find('ZdcName').text else '-8',
                    'ZdcVersion': ecu.find('ZdcVersion').text.strip() if ecu.find('ZdcVersion') is not None and ecu.find('ZdcVersion').text else '-8',
                    'subECU': {},
                    'DTC_info': [],
                    'Ereignisse_count': 0,
                    'Informationen_count': 0,
                    'Aktiv_count': 0,
                    'Passiv_count': 0,
                    'NOCONTACT': ecu.find('NOCONTACT').text.strip() if ecu.find('NOCONTACT') is not None and ecu.find('NOCONTACT').text else '-8'
                }

                SubTeilnehmer = ecu.find('SubTeilnehmer')
                if SubTeilnehmer is not None:
                    sub_ecus = SubTeilnehmer.findall('Sub')
                    for sub_ecu in sub_ecus:
                        sub_hw_address = sub_ecu.find('Adresse').text.strip() if sub_ecu.find('Adresse') is not None and sub_ecu.find('Adresse').text else '-8'
                        ecu_info['subECU'][sub_hw_address] = {
                            'subECU_name': sub_ecu.find('SubtName').text.strip() if sub_ecu.find('SubtName') is not None and sub_ecu.find('SubtName').text else '-8',
                            'subECU_HW_version': sub_ecu.find('HWVersion').text.strip() if sub_ecu.find('HWVersion') is not None and sub_ecu.find('HWVersion').text else '-8',
                            'subECU_HW_partnumber': sub_ecu.find('HWTeilenummer').text.strip() if sub_ecu.find('HWTeilenummer') is not None and sub_ecu.find('HWTeilenummer').text else '-8',
                            'subECU_SW_version': sub_ecu.find('SWVersion').text.strip() if sub_ecu.find('SWVersion') is not None and sub_ecu.find('SWVersion').text else '-8',
                            'subECU_SW_partnumber': sub_ecu.find('SWTeilenummer').text.strip() if sub_ecu.find('SWTeilenummer') is not None and sub_ecu.find('SWTeilenummer').text else '-8',
                            'subECU_Codierung': sub_ecu.find('Codierung').text.strip() if sub_ecu.find('Codierung') is not None and sub_ecu.find('Codierung').text else '-8',
                            'subECU_Schluesseltyp': sub_ecu.find('Schluesseltyp').text.strip() if sub_ecu.find('Schluesseltyp') is not None and sub_ecu.find('Schluesseltyp').text else '-8',
                            'subECU_VkmsStatus': sub_ecu.find('VkmsStatus').text.strip() if sub_ecu.find('VkmsStatus') is not None and sub_ecu.find('VkmsStatus').text else '-8',
                            'subECU_SfdRole': sub_ecu.find('Sfd/SfdRole').text.strip() if sub_ecu.find('Sfd/SfdRole') is not None and sub_ecu.find('Sfd/SfdRole').text else '-8',
                            'subECU_SfdDuration': sub_ecu.find('Sfd/SfdDuration').text.strip() if sub_ecu.find('Sfd/SfdDuration') is not None and sub_ecu.find('Sfd/SfdDuration').text else '-8',
                            'subECU_FAZITString': sub_ecu.find('FAZITString').text.strip() if sub_ecu.find('FAZITString') is not None and sub_ecu.find('FAZITString').text else '-8',
                            'subECU_ZdcName': sub_ecu.find('ZdcName').text.strip() if sub_ecu.find('ZdcName') is not None and sub_ecu.find('ZdcName').text else '-8',
                            'subECU_ZdcVersion': sub_ecu.find('ZdcVersion').text.strip() if sub_ecu.find('ZdcVersion') is not None and sub_ecu.find('ZdcVersion').text else '-8'
                        }

                dtcs = ecu.findall('Fehlerspeicher/Block/Fehler/Fehler')
                if dtcs:
                    for dtc in dtcs:
                        dtc_data = {
                            'DTCNummer_Dec': dtc.find('DTCNummer_Dec').text.strip() if dtc.find('DTCNummer_Dec') is not None and dtc.find('DTCNummer_Dec').text else '-8',
                            'FehlerTyp': dtc.find('FehlerTyp').text.strip() if dtc.find('FehlerTyp') is not None and dtc.find('FehlerTyp').text else '-8',
                            'dtc_odx_text': dtc.find('DTCODXTEXT').text.strip() if dtc.find('DTCODXTEXT') is not None and dtc.find('DTCODXTEXT').text else '-8',
                            'dtc_vas_text': dtc.find('DTCVASTEXT').text.strip() if dtc.find('DTCVASTEXT') is not None and dtc.find('DTCVASTEXT').text else '-8',
                            'odx_error': dtc.find('ODXError').text.strip() if dtc.find('ODXError') is not None and dtc.find('ODXError').text else '-8',
                            'dtc_priority': dtc.find('DtcPriority').text.strip() if dtc.find('DtcPriority') is not None and dtc.find('DtcPriority').text else '-8',
                            'dtc_kind_color': dtc.find('DTCKIND_COLOR').text.strip() if dtc.find('DTCKIND_COLOR') is not None and dtc.find('DTCKIND_COLOR').text else '-8'
                        }
                        ecu_info['DTC_info'].append(dtc_data)

                        if "VKMS" in dtc_data['dtc_odx_text'] or "VKMS" in dtc_data['dtc_vas_text']:
                            ecu_info['isVKMS'] = True
                        elif "SOK" in dtc_data['dtc_odx_text'] or "SOK" in dtc_data['dtc_vas_text']:
                            ecu_info['isSOK'] = True
                        elif "no basic setting" in dtc_data['dtc_odx_text'] or "no basic setting" in dtc_data['dtc_vas_text']:
                            ecu_info['isNoBasicSetting'] = True

                        fault_category = classify_fault(dtc_data['dtc_priority'], dtc_data['dtc_kind_color'])
                        fault_event, fault_status = fault_category
                        if fault_status == 'Aktiv':
                            result['basic_info']['total_dtc_count'] += 1
                            ecu_info['Aktiv_count'] += 1
                            if fault_event == 'Ereignisse':
                                ecu_info['Ereignisse_count'] += 1
                            elif fault_event == 'Informationen':
                                ecu_info['Informationen_count'] += 1
                        elif fault_status == 'Passiv':
                            result['basic_info']['total_dtc_count'] += 1
                            ecu_info['Passiv_count'] += 1

                result['ecu_info'][hw_address] = ecu_info

        return result

    except Exception as e:
        import traceback
        print(f"解析电检报告出错: {e}")
        traceback.print_exc()
        return None
