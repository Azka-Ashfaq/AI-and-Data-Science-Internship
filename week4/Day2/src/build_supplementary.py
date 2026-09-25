"""
Day 2 - Task 1 (cont.): synthetic supplementary knowledge-base tables.
The Kaggle dataset has no amenities/schools/hospitals/payment-plan/FAQ data,
so these are generated deterministically (seeded) per unique location to
round out the knowledge base the brief asks for. Clearly a DEMO/SYNTHETIC
layer -- swap for real company data before production use.
"""
import pandas as pd
import numpy as np
import random
import json

random.seed(7)
np.random.seed(7)

props = pd.read_csv("data/properties_clean.csv")

AMENITY_POOL = [
    "24/7 security", "gated community", "mosque nearby", "park & green belt",
    "underground electricity", "wide carpeted streets", "community center",
    "gym & fitness club", "swimming pool (society level)", "commercial market nearby",
    "backup generator in common areas", "children's play area",
]
SCHOOL_POOL = [
    "Beaconhouse School System", "The City School", "Lahore Grammar School",
    "Roots International Schools", "Bahria College", "Froebel's International",
    "Aitchison College", "Divisional Public School", "Convent of Jesus & Mary",
]
HOSPITAL_POOL = [
    "Shifa International Hospital", "Doctors Hospital", "South City Hospital",
    "Aga Khan University Hospital", "Services Hospital", "Fauji Foundation Hospital",
    "Ittefaq Hospital", "PIMS Hospital",
]
DEVELOPERS = {
    "DHA Defence": "DHA City Developer Authority", "Bahria Town": "Bahria Town Pvt. Ltd.",
    "Johar Town": "LDA (Lahore Development Authority)", "Gulberg": "Gulberg Greens Developers",
}

locations = props[["city", "location"]].drop_duplicates().reset_index(drop=True)

amenities_rows, schools_rows, hospitals_rows, plans_rows = [], [], [], []
for _, r in locations.iterrows():
    city, loc = r["city"], r["location"]
    amens = random.sample(AMENITY_POOL, k=random.randint(4, 7))
    amenities_rows.append({"city": city, "location": loc, "amenities": ", ".join(amens)})

    schools = random.sample(SCHOOL_POOL, k=random.randint(1, 3))
    for s in schools:
        schools_rows.append({"city": city, "location": loc, "school_name": s,
                              "distance_km": round(random.uniform(0.5, 4.5), 1)})

    hospitals = random.sample(HOSPITAL_POOL, k=random.randint(1, 2))
    for h in hospitals:
        hospitals_rows.append({"city": city, "location": loc, "hospital_name": h,
                                "distance_km": round(random.uniform(0.8, 6.0), 1)})

    developer = DEVELOPERS.get(loc, f"{loc} Developers Pvt. Ltd.")
    down_pct = random.choice([10, 15, 20, 25])
    years = random.choice([2, 3, 4, 5])
    plans_rows.append({
        "city": city, "location": loc, "developer": developer,
        "down_payment_pct": down_pct, "installment_years": years,
        "plan_summary": f"{down_pct}% down payment, remaining balance in {years}-year "
                         f"quarterly installments through {developer}."
    })

pd.DataFrame(amenities_rows).to_csv("data/amenities.csv", index=False)
pd.DataFrame(schools_rows).to_csv("data/schools.csv", index=False)
pd.DataFrame(hospitals_rows).to_csv("data/hospitals.csv", index=False)
pd.DataFrame(plans_rows).to_csv("data/payment_plans.csv", index=False)

faqs = [
    ("What is the minimum down payment for an installment plan?",
     "Down payment ranges 10-25% of the total price depending on the developer and society; RealEstate Hub will confirm the exact figure for the specific project."),
    ("Do you charge a commission to buyers?",
     "RealEstate Hub charges no fee to buyers or renters on completed deals; our commission is paid by the seller/landlord per standard industry practice."),
    ("Can I visit a property before booking?",
     "Yes, every listing can be visited in person; our agent will accompany you and can be scheduled via a phone call or the booking flow."),
    ("Is possession immediate after full payment?",
     "For ready properties, possession is handed over within 7-15 working days of full payment and paperwork completion. Under-construction units follow the developer's handover schedule."),
    ("What documents are required to book a property?",
     "A valid CNIC, proof of payment for the token/down payment, and a signed booking form are required to secure a property."),
    ("Do you help with rental agreements?",
     "Yes, RealEstate Hub drafts and facilitates signing of the tenancy agreement for all rental bookings made through us."),
    ("Are the prices negotiable?",
     "Most listings have some flexibility; our agent will share the realistic negotiable range once you express serious interest in a specific property."),
    ("What happens if I want to cancel a booked visit?",
     "You can cancel or reschedule a visit any time before the appointment by calling us back; there is no penalty for cancelling a visit."),
    ("Do you handle commercial properties?",
     "Yes, we list commercial plots, shops, and offices in addition to residential houses, flats, and portions."),
    ("How is the appreciation potential of a society assessed?",
     "We look at historical price trends, infrastructure development, and developer track record for that society; our agent can share specific comparables on request."),
    ("Can NRI/overseas Pakistanis book a property remotely?",
     "Yes, bookings can be completed remotely with power-of-attorney or verified digital documentation; a dedicated agent assists overseas clients."),
    ("What is the difference between a Marla and a Kanal?",
     "1 Kanal equals 20 Marla; both are traditional South Asian land area units used throughout property listings in Pakistan."),
    ("Do you offer investment consultation, not just listings?",
     "Yes, for investment inquiries we discuss budget, ROI goals, and time horizon before recommending specific plots or units."),
    ("Is maintenance handled by the society or the buyer?",
     "Ongoing society maintenance (security, common areas) is typically covered by a monthly society fee; unit-level maintenance is the owner's responsibility."),
    ("How quickly can an appointment be booked?",
     "Most visit appointments can be scheduled for the same day or the next day, subject to agent and property availability."),
]
pd.DataFrame(faqs, columns=["question", "answer"]).to_csv("data/faqs.csv", index=False)

print("Supplementary tables written:")
for f in ["amenities", "schools", "hospitals", "payment_plans", "faqs"]:
    d = pd.read_csv(f"data/{f}.csv")
    print(f"  {f}.csv -> {len(d)} rows")
