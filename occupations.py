"""Rule-based mapping of free-text job titles to occupation families.

Order matters: the first matching pattern wins, so more specific patterns
come before generic ones (e.g. "nurse practitioner" before "nurse").
"""
import re

FAMILIES = [
    ("Physician / Surgeon",          r"\b(physician|surgeon|hospitalist|psychiatrist|anesthesiologist|radiologist|\bmd\b)"),
    ("Nurse Practitioner / PA",      r"nurse practitioner|\bnp\b|physician assistant|\bpa-c\b"),
    ("Pharmacist",                   r"pharmacist"),
    ("Pharmacy Technician",          r"pharmacy tech"),
    ("Physical / Occupational Therapist", r"physical therapist|occupational therapist|speech.{0,10}patholog|\bslp\b|\bpta\b|therapy assistant"),
    ("Registered Nurse",             r"registered nurse|\brn\b|nurse"),
    ("Medical Assistant / CNA",      r"medical assistant|\bcna\b|nursing assistant|patient care tech|phlebotom|caregiver|home health aide|caretaker|\bdsp\b|direct support"),
    ("Behavior Technician / Counselor", r"behavior tech|behavior interventionist|\brbt\b|\bbcba\b|counselor|social worker|therapist|case manager"),
    ("Dental",                       r"dental|dentist|hygienist"),
    ("Teacher / Educator",           r"teacher|tutor|instructor|professor|educator|lecturer|paraprofessional|preschool"),
    ("Data Scientist / ML",          r"data scien|machine learning|\bml\b|\bai\b|research scientist|applied scientist"),
    ("Data / BI Analyst",            r"data analyst|business intelligence|\bbi\b|analytics|data engineer"),
    ("Software Engineer",            r"software|developer|\bsde\b|programmer|front.?end|back.?end|full.?stack|devops|site reliability|\bsre\b|web dev|ios|android"),
    ("IT / Systems / Security",      r"\bit\b|help ?desk|desktop support|system(s)? admin|network|cyber|security engineer|information security|technical support|it support|cloud"),
    ("Product Manager",              r"product manager|product owner|product lead"),
    ("Designer (UX/Graphic)",        r"designer|\bux\b|\bui\b|graphic|creative director|art director"),
    ("Engineer (non-software)",      r"engineer"),
    ("Scientist / Lab",              r"scientist|lab tech|laboratory|chemist|biolog|research associate"),
    ("Accountant / Auditor",         r"accountant|accounting|auditor|audit|\bcpa\b|tax |controller|bookkeep"),
    ("Financial Analyst / Banking",  r"financial analyst|finance|investment|banker|banking|actuar|underwriter|credit analyst|\bfp&a\b|wealth|financial advisor"),
    ("Bank Teller / Loan Officer",   r"teller|loan officer|mortgage"),
    ("Consultant",                   r"consultant|consulting|advisory"),
    ("HR / Recruiter",               r"recruit|human resources|\bhr\b|talent|people partner|payroll"),
    ("Marketing / Communications",   r"marketing|brand|communications|content|social media|seo|public relations|copywriter"),
    ("Account Executive / B2B Sales",r"account executive|business development|\bbdr\b|\bsdr\b|sales development|account manager|sales representative|sales rep|sales manager|sales executive|territory|inside sales|outside sales|sales engineer"),
    ("Project / Program Manager",    r"project manager|program manager|project coordinator|scrum|delivery manager|pmo"),
    ("Business / Operations Analyst",r"business analyst|operations analyst|analyst"),
    ("Operations / General Manager", r"operations manager|general manager|director|vice president|\bvp\b|head of|chief|plant manager|operations"),
    ("Administrative / Office",      r"administrative|admin|receptionist|front desk|office|secretary|clerk|executive assistant|coordinator|scheduler|data entry"),
    ("Customer Service",             r"customer service|customer support|call center|customer care|customer success|client service|support specialist|guest service|member service"),
    ("Legal",                        r"attorney|lawyer|paralegal|legal|counsel"),
    ("Retail Store Manager",         r"store manager|assistant manager|shift manager|shift lead|shift supervisor|department manager|key ?holder|team lead"),
    ("Cashier / Retail Associate",   r"cashier|sales associate|retail|merchandis|stocker|personal shopper|sales floor|store associate|team member|crew member|crew|associate"),
    ("Barista",                      r"barista"),
    ("Cook / Kitchen",               r"cook|chef|kitchen|dishwasher|prep|baker|food prep"),
    ("Server / Bartender / Host",    r"server|bartender|host|busser|waiter|waitress|food runner|food service|restaurant"),
    ("Housekeeping / Janitorial",    r"housekeep|janitor|custodian|cleaner|cleaning|porter|environmental services"),
    ("Driver / Delivery",            r"driver|delivery|courier|cdl|truck"),
    ("Warehouse / Logistics",        r"warehouse|forklift|picker|packer|package handler|material handler|logistics|shipping|receiving|freight|fulfillment|inventory|supply chain"),
    ("Maintenance / Skilled Trades", r"maintenance|technician|mechanic|electrician|plumber|hvac|carpenter|welder|machinist|installer|repair|construction|laborer|painter"),
    ("Manufacturing / Production",   r"production|manufactur|assembl|operator|fabricat|quality"),
    ("Security Guard",               r"security officer|security guard|loss prevention|asset protection|security"),
    ("Real Estate / Property",       r"real estate|leasing|property manager|realtor"),
    ("Intern",                       r"intern\b|internship"),
    ("Manager (general)",            r"manager|supervisor|lead"),
]
_COMPILED = [(name, re.compile(p)) for name, p in FAMILIES]


def map_title(title):
    if not isinstance(title, str):
        return None
    t = title.lower()
    for name, rx in _COMPILED:
        if rx.search(t):
            return name
    return None
