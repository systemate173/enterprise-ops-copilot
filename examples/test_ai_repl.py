from src.ai_summary import ai_manager_summary

det = {
  "ready": True,
  "summary": "Engineering incident with High urgency.",
  "highlights": ["Check CI/CD logs", "Confirm environment"],
  "sources": ["RBK-ENG-CICD-101"],
}

print(ai_manager_summary(det))
