# TODO — Yassir RH (AI Recruitment SaaS)

## Plan (validation utilisateur OK)
1) Routes/templating manquants : Statistiques, Settings, AI Assistant + KPIs dashboard réels depuis PostgreSQL.
2) Inscription conforme : OTP/validation email + (éventuel) schéma DB + template de vérification.
3) RTL/i18n et dark/light (correctif si nécessaire).
4) Validation UX matching/upload (drag&drop, multi-upload, OCR fallback) et cohérence affichage résultats.

---

## Étapes
- [ ] 1.1 Inspecter `shema.sql` + colonnes DB actuelles (users/offres/cvs) pour brancher KPIs.
- [ ] 1.2 Ajouter endpoints Flask + templates : `/statistiques`, `/settings`, `/assistant-ia`.
- [ ] 1.3 Mettre à jour `templates/dashboard.html` pour utiliser valeurs DB (offres, cvs, avg score, activité récente).
- [ ] 1.4 Ajouter menu sidebar vers ces nouveaux écrans si manquant (déjà présent partiellement).

- [ ] 2.1 Mettre en place OTP backend : routes `/register`, `/verify-otp` (+ éventuel `/resend-otp`).
- [ ] 2.2 Ajuster schéma DB (users + champs OTP + champs avatar).
- [ ] 2.3 Ajouter template `templates/verify_otp.html`.
- [ ] 2.4 Adapter `templates/register.html` pour OTP workflow (submission step par step ou submit final + message OTP).
- [ ] 2.5 Empêcher login tant que email non validé.

- [ ] 3.1 RTL : appliquer `dir="rtl"` et ajustements CSS quand langue = ar.
- [ ] 3.2 Harmoniser thème (landing + dashboard) côté stockage (localStorage) si besoin.

- [ ] 4.1 Vérifier `templates/resultats.html` : variables (`résultats` vs `candidats`) cohérentes avec `app.py`.
- [ ] 4.2 Lancer tests manuels du workflow complet.

