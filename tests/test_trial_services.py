import unittest

from services.embeddings import build_embedding_text
from services.trial_documents import extract_essential_fields, format_study
from services.trial_filters import build_query_from_filters, get_semantic_query_text


class TrialServiceTests(unittest.TestCase):
    def test_build_query_from_filters_supports_nested_fields(self):
        query = build_query_from_filters(
            {
                "condition": "diabetes",
                "status": ["RECRUITING"],
                "phase": ["PHASE2"],
                "sponsor": "Novo",
            }
        )

        self.assertIn("$and", query)
        clauses = query["$and"]
        self.assertTrue(any("protocolSection.conditionsModule.conditions" in str(clause) for clause in clauses))
        self.assertTrue(any("protocolSection.statusModule.overallStatus" in str(clause) for clause in clauses))
        self.assertTrue(any("protocolSection.designModule.phases" in str(clause) for clause in clauses))
        self.assertTrue(any("protocolSection.sponsorCollaboratorsModule.leadSponsor.name" in str(clause) for clause in clauses))

    def test_get_semantic_query_text_prefers_explicit_query(self):
        semantic_text = get_semantic_query_text(
            {
                "query": "heart attack",
                "condition": "myocardial infarction",
                "title": "cardiology",
            }
        )
        self.assertEqual(semantic_text, "heart attack")

    def test_format_study_uses_nested_document_shape(self):
        study = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT123", "briefTitle": "Test Trial"},
                "statusModule": {"overallStatus": "RECRUITING"},
                "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE2"]},
                "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Acme Bio"}},
            },
            "hasResults": True,
        }

        formatted = format_study(study)
        self.assertEqual(formatted["protocolSection"]["identificationModule"]["nctId"], "NCT123")
        self.assertEqual(formatted["protocolSection"]["designModule"]["phases"], ["PHASE2"])
        self.assertTrue(formatted["hasResults"])

    def test_extract_essential_fields_supports_flat_documents(self):
        study = {
            "nct_id": "NCT999",
            "title": "Flat Trial",
            "status": "COMPLETED",
            "studyType": "OBSERVATIONAL",
            "phase": "NA",
            "conditions": ["Rare Disease"],
            "interventions": ["Drug A"],
            "primaryOutcome": "Symptom reduction",
            "sponsor": "Example Sponsor",
            "enrollment": 120,
            "sex": "ALL",
            "minimumAge": "18 Years",
            "maximumAge": "65 Years",
            "countries": ["United States"],
        }

        extracted = extract_essential_fields(study)
        self.assertEqual(extracted["nctId"], "NCT999")
        self.assertEqual(extracted["interventions"], ["Drug A"])
        self.assertEqual(extracted["eligibility"]["ageRange"], "18 Years to 65 Years")

    def test_build_embedding_text_handles_nested_trials(self):
        study = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT777", "briefTitle": "Nested Trial"},
                "conditionsModule": {"conditions": ["Oncology"]},
                "armsInterventionsModule": {"interventions": [{"name": "Drug B"}]},
                "descriptionModule": {"briefSummary": "Short summary"},
                "statusModule": {"overallStatus": "ACTIVE_NOT_RECRUITING"},
            }
        }

        text = build_embedding_text(study)
        self.assertIn("NCT ID: NCT777", text)
        self.assertIn("Conditions: Oncology", text)
        self.assertIn("Interventions: Drug B", text)


if __name__ == "__main__":
    unittest.main()
