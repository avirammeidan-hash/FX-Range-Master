# CI/CD Setup — GitHub Secrets Required

The workflow in `.github/workflows/deploy.yml` needs **three repository secrets**
set in **GitHub → Settings → Secrets and variables → Actions**.

---

## 1. `GCP_SA_KEY`  
A Google Cloud service-account JSON key that has permission to deploy Cloud Run.

### Create the service account (one-time):
```bash
# Create SA
gcloud iam service-accounts create github-deployer \
  --project fx-range-master \
  --display-name "GitHub Actions deployer"

# Grant roles
gcloud projects add-iam-policy-binding fx-range-master \
  --member "serviceAccount:github-deployer@fx-range-master.iam.gserviceaccount.com" \
  --role "roles/run.admin"

gcloud projects add-iam-policy-binding fx-range-master \
  --member "serviceAccount:github-deployer@fx-range-master.iam.gserviceaccount.com" \
  --role "roles/storage.admin"

gcloud projects add-iam-policy-binding fx-range-master \
  --member "serviceAccount:github-deployer@fx-range-master.iam.gserviceaccount.com" \
  --role "roles/cloudbuild.builds.builder"

gcloud projects add-iam-policy-binding fx-range-master \
  --member "serviceAccount:github-deployer@fx-range-master.iam.gserviceaccount.com" \
  --role "roles/iam.serviceAccountUser"

# Download the key
gcloud iam service-accounts keys create /tmp/gcp-sa-key.json \
  --iam-account github-deployer@fx-range-master.iam.gserviceaccount.com
```

Add the **entire contents** of `/tmp/gcp-sa-key.json` as the `GCP_SA_KEY` secret.

---

## 2. `FIREBASE_SA_JSON`  
The `firebase-service-account.json` file content (already on your machine, excluded from git).

```bash
# On your local machine:
cat C:\workgit\FX-Range-Master\firebase-service-account.json
```

Paste the entire JSON as the `FIREBASE_SA_JSON` secret.

---

## 3. (Already configured above — no third secret needed)

The project ID, service name, and region are hard-coded in the workflow file
(`fx-range-master` / `me-west1`), so no extra secret is needed for those.

---

## What the pipeline does

| Step | What happens |
|------|-------------|
| **Lint** | Python syntax check + TypeScript type check |
| **Build frontend** | `npm ci && npm run build` in `frontend/` |
| **Deploy** | `gcloud run deploy --source .` (Cloud Build builds image, Cloud Run serves it) |

Every push to `main` triggers a full deploy. You can also trigger manually from
**GitHub → Actions → Deploy to Cloud Run → Run workflow**.
