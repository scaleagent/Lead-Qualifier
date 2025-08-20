
# modules/qualification/contractor_profiles.py

# Master list of all possible qualification data categories
MASTER_DATA_CATEGORIES = {
    # Basic job information (universal)
    "job_type": {
        "required": True,
        "prompt": "What type of work do you need done?",
        "description": "Type of work or service required"
    },
    "property_type": {
        "required": True,
        "prompt": "What type of property is this for?",
        "description": "Residential, commercial, industrial, etc."
    },
    "urgency": {
        "required": True,
        "prompt": "How urgent is this work?",
        "description": "Timeline or urgency level"
    },
    "address": {
        "required": True,
        "prompt": "What's the property address?",
        "description": "Full property address"
    },
    "access": {
        "required": True,
        "prompt": "How can we access the property?",
        "description": "Access arrangements and availability"
    },
    
    # Property details
    "property_age": {
        "required": False,
        "prompt": "How old is the property?",
        "description": "Age of building/property"
    },
    "property_size": {
        "required": False,
        "prompt": "What's the size of the property?",
        "description": "Square footage, number of rooms, etc."
    },
    "property_floors": {
        "required": False,
        "prompt": "How many floors does the property have?",
        "description": "Number of floors/levels"
    },
    "property_occupancy": {
        "required": False,
        "prompt": "Is the property currently occupied?",
        "description": "Occupancy status during work"
    },
    
    # Electrical specific
    "electrical_system_age": {
        "required": False,
        "prompt": "How old is the electrical system?",
        "description": "Age of existing electrical installation"
    },
    "electrical_panel_type": {
        "required": False,
        "prompt": "What type of electrical panel do you have?",
        "description": "Fuse box, circuit breaker type, etc."
    },
    "electrical_issue_location": {
        "required": False,
        "prompt": "Where exactly is the electrical issue?",
        "description": "Specific location of electrical problem"
    },
    "power_outage_history": {
        "required": False,
        "prompt": "Any recent power outages or electrical issues?",
        "description": "History of electrical problems"
    },
    
    # Plumbing specific
    "water_system_type": {
        "required": False,
        "prompt": "What type of water system do you have?",
        "description": "Mains water, well water, etc."
    },
    "plumbing_age": {
        "required": False,
        "prompt": "How old is the plumbing system?",
        "description": "Age of existing plumbing"
    },
    "water_pressure": {
        "required": False,
        "prompt": "Any issues with water pressure?",
        "description": "Water pressure problems"
    },
    "leak_location": {
        "required": False,
        "prompt": "Where is the leak located?",
        "description": "Specific location of water leak"
    },
    
    # Roofing specific
    "roof_type": {
        "required": False,
        "prompt": "What type of roof do you have?",
        "description": "Tile, slate, flat, pitched, etc."
    },
    "roof_age": {
        "required": False,
        "prompt": "How old is the roof?",
        "description": "Age of existing roof"
    },
    "roof_area": {
        "required": False,
        "prompt": "What's the approximate roof area?",
        "description": "Size of roof requiring work"
    },
    "roof_access": {
        "required": False,
        "prompt": "How accessible is the roof?",
        "description": "Roof access conditions"
    },
    
    # HVAC specific
    "heating_system_type": {
        "required": False,
        "prompt": "What type of heating system do you have?",
        "description": "Gas, electric, oil, heat pump, etc."
    },
    "cooling_system_type": {
        "required": False,
        "prompt": "What type of cooling system do you have?",
        "description": "Central air, window units, etc."
    },
    "hvac_age": {
        "required": False,
        "prompt": "How old is your HVAC system?",
        "description": "Age of heating/cooling system"
    },
    "temperature_issues": {
        "required": False,
        "prompt": "What temperature issues are you experiencing?",
        "description": "Heating/cooling problems"
    },
    
    # Construction/General
    "building_materials": {
        "required": False,
        "prompt": "What materials is the property built with?",
        "description": "Brick, timber frame, concrete, etc."
    },
    "structural_changes": {
        "required": False,
        "prompt": "Are any structural changes needed?",
        "description": "Walls, supports, foundations"
    },
    "planning_permission": {
        "required": False,
        "prompt": "Do you need planning permission for this work?",
        "description": "Planning permission requirements"
    },
    
    # Budget and timeline
    "budget_range": {
        "required": False,
        "prompt": "What's your budget range for this work?",
        "description": "Expected budget or price range"
    },
    "preferred_schedule": {
        "required": False,
        "prompt": "When would you prefer the work to be done?",
        "description": "Preferred timing/schedule"
    },
    "work_duration": {
        "required": False,
        "prompt": "How long do you expect the work to take?",
        "description": "Expected duration of work"
    },
    
    # Insurance and compliance
    "insurance_claim": {
        "required": False,
        "prompt": "Is this work for an insurance claim?",
        "description": "Insurance claim involvement"
    },
    "safety_concerns": {
        "required": False,
        "prompt": "Are there any safety concerns we should know about?",
        "description": "Safety considerations"
    },
    
    # Contact and follow-up
    "best_contact_time": {
        "required": False,
        "prompt": "What's the best time to contact you?",
        "description": "Preferred contact times"
    },
    "alternative_contact": {
        "required": False,
        "prompt": "Is there an alternative contact number?",
        "description": "Secondary contact information"
    }
}

# Contractor profile definitions - each profile activates specific categories
CONTRACTOR_PROFILES = {
    "electrician": {
        "name": "Electrician",
        "active_categories": [
            # Universal required
            "job_type", "property_type", "urgency", "address", "access",
            # Property details
            "property_age", "property_size", "property_occupancy",
            # Electrical specific
            "electrical_system_age", "electrical_panel_type", "electrical_issue_location", "power_outage_history",
            # General
            "budget_range", "preferred_schedule", "best_contact_time", "safety_concerns"
        ],
        "personality": "Professional electrical contractor's assistant with technical knowledge of electrical systems and safety requirements."
    },
    
    "plumber": {
        "name": "Plumber", 
        "active_categories": [
            # Universal required
            "job_type", "property_type", "urgency", "address", "access",
            # Property details
            "property_age", "property_size", "property_floors",
            # Plumbing specific
            "water_system_type", "plumbing_age", "water_pressure", "leak_location",
            # General
            "budget_range", "preferred_schedule", "best_contact_time", "insurance_claim"
        ],
        "personality": "Experienced plumbing contractor's assistant familiar with water systems and plumbing installations."
    },
    
    "roofer": {
        "name": "Roofer",
        "active_categories": [
            # Universal required
            "job_type", "property_type", "urgency", "address", "access",
            # Property details
            "property_age", "property_size",
            # Roofing specific
            "roof_type", "roof_age", "roof_area", "roof_access",
            # General
            "building_materials", "planning_permission", "budget_range", "preferred_schedule", 
            "insurance_claim", "best_contact_time", "safety_concerns"
        ],
        "personality": "Knowledgeable roofing contractor's assistant with expertise in roof types, materials, and weather protection."
    },
    
    "hvac": {
        "name": "HVAC Technician",
        "active_categories": [
            # Universal required
            "job_type", "property_type", "urgency", "address", "access",
            # Property details
            "property_age", "property_size", "property_floors",
            # HVAC specific
            "heating_system_type", "cooling_system_type", "hvac_age", "temperature_issues",
            # General
            "budget_range", "preferred_schedule", "best_contact_time"
        ],
        "personality": "Expert HVAC contractor's assistant specializing in heating, ventilation, and air conditioning systems."
    },
    
    "general_contractor": {
        "name": "General Contractor",
        "active_categories": [
            # Universal required
            "job_type", "property_type", "urgency", "address", "access",
            # Property details
            "property_age", "property_size", "property_floors", "property_occupancy",
            # Construction specific
            "building_materials", "structural_changes", "planning_permission",
            # General
            "budget_range", "preferred_schedule", "work_duration", "best_contact_time", 
            "insurance_claim", "safety_concerns"
        ],
        "personality": "Experienced general contractor's assistant with broad construction knowledge and project management skills."
    }
}

def get_contractor_profile(profile_type: str) -> dict:
    """Get contractor profile configuration by type."""
    return CONTRACTOR_PROFILES.get(profile_type, CONTRACTOR_PROFILES["general_contractor"])

def get_active_categories_for_profile(profile_type: str) -> list:
    """Get list of active data categories for a contractor profile."""
    profile = get_contractor_profile(profile_type)
    return profile["active_categories"]

def get_category_config(category_name: str) -> dict:
    """Get configuration for a specific data category."""
    return MASTER_DATA_CATEGORIES.get(category_name, {})

def get_required_categories_for_profile(profile_type: str) -> list:
    """Get list of required categories for a contractor profile."""
    active_categories = get_active_categories_for_profile(profile_type)
    return [cat for cat in active_categories if MASTER_DATA_CATEGORIES.get(cat, {}).get("required", False)]
