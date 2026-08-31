"""
LLM-powered CRM data generator for realistic Sri Lankan healthcare data.

Uses OpenAI API to generate culturally appropriate names, dispute resolution notes, and reasons.
"""

import json
from loguru import logger
from typing import List, Dict
from pathlib import Path
class HealthcareDataGenerator:
    """Generate realistic healthcare data using LLM."""
    
    def __init__(self, llm):
        """
        Initialize generator with LLM.
        
        Args:
            llm: Language model instance (from get_chat_llm)
        """
        self.llm = llm
        self._cache = {}
    
    def generate_doctors(self, n: int, specialties: List[str]) -> List[Dict]:
        """
        Generate doctor data with Sri Lankan names.
        
        Args:
            n: Number of doctors to generate
            specialties: List of specialty names to assign
            
        Returns:
            List of doctor dicts with name, specialty, contact info
        """
        cache_key = f"doctors_{n}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        logger.info(f"Generating {n} Sri Lankan doctor names via LLM...")
        
        prompt = f"""Generate {n} realistic Sri Lankan doctor names for a store system.

Requirements:
- Mix of Sinhala and Tamil names
- Include appropriate titles (Dr.)
- Mix of male and female names
- Professional medical names
- NO fictional or duplicate names

Output as JSON array:
[
  {{"full_name": "Dr. Priya Fernando", "gender": "F"}},
  {{"full_name": "Dr. Rohan Silva", "gender": "M"}},
  ...
]

Generate exactly {n} doctors:"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                doctors_data = json.loads(json_str)
            else:
                raise ValueError("No JSON array found in response")
            
            # Assign specialties cyclically
            doctors = []
            for i, doc_data in enumerate(doctors_data[:n]):
                doctors.append({
                    "full_name": doc_data["full_name"],
                    "gender": doc_data.get("gender", "M"),
                    "specialty": specialties[i % len(specialties)],
                })
            
            self._cache[cache_key] = doctors
            logger.info(f"✓ Generated {len(doctors)} doctor names")
            return doctors
            
        except Exception as e:
            logger.error(f"Failed to generate doctor names: {e}")
            logger.warning("Falling back to template names...")
            return self._fallback_doctors(n, specialties)
    
    def generate_patients(self, n: int) -> List[Dict]:
        """
        Generate patient data with Sri Lankan names.
        
        Args:
            n: Number of patients to generate
            
        Returns:
            List of patient dicts with name, gender, DOB
        """
        cache_key = f"patients_{n}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        logger.info(f"Generating {n} Sri Lankan patient names via LLM...")
        
        prompt = f"""Generate {n} realistic Sri Lankan patient names for a store database.

Requirements:
- Mix of Sinhala and Tamil names
- Mix of male and female names
- Include children, adults, and elderly
- Culturally appropriate names
- NO fictional or duplicate names

Output as JSON array:
[
  {{"full_name": "Anushka Perera", "gender": "F", "dob": "1985-03-15"}},
  {{"full_name": "Kamal Jayasuriya", "gender": "M", "dob": "1972-08-22"}},
  {{"full_name": "Nethmi Wijesinghe", "gender": "F", "dob": "2010-11-05"}},
  ...
]

Generate exactly {n} patients with realistic DOBs (YYYY-MM-DD):"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                patients_data = json.loads(json_str)
            else:
                raise ValueError("No JSON array found in response")
            
            patients = patients_data[:n]
            
            self._cache[cache_key] = patients
            logger.info(f"✓ Generated {len(patients)} patient names")
            return patients
            
        except Exception as e:
            logger.error(f"Failed to generate patient names: {e}")
            logger.warning("Falling back to template names...")
            return self._fallback_patients(n)
    
    def generate_appointment_reasons(self, n: int, specialty: str) -> List[str]:
        """
        Generate realistic appointment reasons for a specialty.
        
        Args:
            n: Number of reasons to generate
            specialty: Dispute type
            
        Returns:
            List of appointment reason strings
        """
        cache_key = f"reasons_{specialty}_{n}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        logger.info(f"Generating {n} appointment reasons for {specialty}...")
        
        prompt = f"""Generate {n} realistic dispute claim reasons for {specialty}.

Requirements:
- Professional medical terminology
- Mix of common and specific conditions
- Appropriate for {specialty}
- Brief and clear (5-10 words each)

Output as JSON array of strings:
["Chest pain and shortness of breath", "Annual cardiac checkup", ...]

Generate exactly {n} reasons:"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                reasons = json.loads(json_str)
            else:
                raise ValueError("No JSON array found in response")
            
            reasons = reasons[:n]
            
            self._cache[cache_key] = reasons
            logger.info(f"✓ Generated {len(reasons)} appointment reasons")
            return reasons
            
        except Exception as e:
            logger.error(f"Failed to generate reasons: {e}")
            return self._fallback_reasons(n, specialty)
    
    def generate_medical_notes(self, n: int) -> List[str]:
        """
        Generate realistic dispute resolution notes for patients.
        
        Args:
            n: Number of notes to generate
            
        Returns:
            List of medical note strings
        """
        cache_key = f"notes_{n}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        logger.info(f"Generating {n} dispute resolution notes via LLM...")
        
        prompt = f"""Generate {n} brief dispute resolution notes for patient records.

Requirements:
- Professional medical language
- Mix of conditions (diabetes, hypertension, allergies, medications)
- Realistic but anonymized
- 1-2 sentences each

Output as JSON array of strings:
["Type 2 diabetes, on metformin 500mg BD. Regular monitoring required.", ...]

Generate exactly {n} notes:"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                notes = json.loads(json_str)
            else:
                raise ValueError("No JSON array found in response")
            
            notes = notes[:n]
            
            self._cache[cache_key] = notes
            logger.info(f"✓ Generated {len(notes)} dispute resolution notes")
            return notes
            
        except Exception as e:
            logger.error(f"Failed to generate notes: {e}")
            return self._fallback_notes(n)
    
    # Fallback methods (if LLM fails)
    
    def _fallback_doctors(self, n: int, specialties: List[str]) -> List[Dict]:
        """Fallback doctor names if LLM fails."""
        base_names = [
            ("Dr. Priya Fernando", "F"),
            ("Dr. Rohan Silva", "M"),
            ("Dr. Amara Perera", "F"),
            ("Dr. Nimal Jayasuriya", "M"),
            ("Dr. Sanduni Wijesinghe", "F"),
            ("Dr. Kasun Rajapaksa", "M"),
            ("Dr. Chamari De Silva", "F"),
            ("Dr. Dinesh Wickramasinghe", "M"),
            ("Dr. Tharini Gunawardena", "F"),
            ("Dr. Saman Bandara", "M"),
        ]
        
        doctors = []
        for i in range(n):
            name, gender = base_names[i % len(base_names)]
            if i >= len(base_names):
                name = f"{name} {i+1}"
            doctors.append({
                "full_name": name,
                "gender": gender,
                "specialty": specialties[i % len(specialties)],
            })
        
        return doctors
    
    def _fallback_patients(self, n: int) -> List[Dict]:
        """Fallback patient names if LLM fails."""
        base_names = [
            ("Anushka Perera", "F", "1985-03-15"),
            ("Kamal Jayasuriya", "M", "1972-08-22"),
            ("Nethmi Wijesinghe", "F", "2010-11-05"),
            ("Sunil Fernando", "M", "1968-12-30"),
            ("Madhavi Silva", "F", "1995-06-18"),
        ]
        
        patients = []
        for i in range(n):
            name, gender, dob = base_names[i % len(base_names)]
            if i >= len(base_names):
                name = f"{name} {i+1}"
            patients.append({
                "full_name": name,
                "gender": gender,
                "dob": dob,
            })
        
        return patients
    
    def _fallback_reasons(self, n: int, specialty: str) -> List[str]:
        """Fallback appointment reasons if LLM fails."""
        reasons = [
            f"Routine {specialty.lower()} consultation",
            f"{specialty} follow-up visit",
            f"Annual {specialty.lower()} checkup",
            "Diagnostic consultation",
            "Treatment review",
        ]
        return [reasons[i % len(reasons)] for i in range(n)]
    
    def _fallback_notes(self, n: int) -> List[str]:
        """Fallback dispute resolution notes if LLM fails."""
        notes = [
            "No known allergies. Regular checkups recommended.",
            "Hypertension, on medication. Monitor BP regularly.",
            "Diabetes mellitus type 2. Diet-controlled.",
            "Asthma, uses inhaler PRN.",
            "No significant medical history.",
        ]
        return [notes[i % len(notes)] for i in range(n)]


def get_data_generator():
    """
    Get singleton data generator instance.
    
    Returns:
        HealthcareDataGenerator with LLM
    """
    from infrastructure.llm import get_chat_llm
    
    llm = get_chat_llm()
    return HealthcareDataGenerator(llm)

