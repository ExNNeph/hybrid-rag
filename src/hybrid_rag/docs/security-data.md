# Security and data protection

## Where the data is stored

All customer data is stored in data centres in Frankfurt am Main, Germany (cloud region eu-central-1).
Backups are kept in a second data centre in the same region. We do not transfer customer data outside the
European Union. Our sub-processors are listed at talvi.example/subprocessors.

## Encryption

Data is encrypted in transit with TLS 1.2 or higher and at rest with AES-256. Encryption keys are managed
in a key management service and rotated every 12 months. File attachments are encrypted with a separate
key per workspace.

## Backups

We back up all workspaces every 6 hours and keep the backups for 35 days. A single deleted board can be
restored by the customer from the trash for 30 days. A full workspace restore from a backup is available on
request for Business and Enterprise customers and takes up to 2 business days.

## Single sign-on and two-factor authentication

Single sign-on with SAML 2.0 is available on Business and Enterprise; it works with common identity
providers. Automatic user provisioning with SCIM is Enterprise only. Every user can turn on two-factor
authentication with an authenticator app, and admins on Business and Enterprise can make it mandatory.

## Data deletion after cancellation

When a workspace owner deletes the workspace, all boards and files are deleted within 30 days, and
backups containing them expire after a further 35 days. A data export (CSV or JSON) can be downloaded
before deletion under Settings > Export. Deleted workspaces cannot be recovered.

## Data processing agreement

A data processing agreement (DPA) under Art. 28 GDPR is part of our terms for all paid plans and can be
downloaded as a signed PDF under Settings > Legal. Enterprise customers can negotiate individual changes.

## Reporting a vulnerability

Security researchers can report vulnerabilities to security@talvi.example. We confirm receipt within
2 business days and do not take legal action against good-faith research. We do not run a paid bug bounty.
