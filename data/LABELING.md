# Taskline Support Email Labeling Guide

## Purpose

This guide gives the classifier's four categories precise, written definitions so that dataset labels and prompt behavior can be checked against the same rule instead of two different guesses.

The guiding principle: **the correct label is the team that resolves the email.** Do not label by keyword. Ask which team would actually pick up the ticket and fix the thing the customer needs fixed.

## Billing

Definition: charges, refunds, invoices, payment methods, purchase orders, pricing, plan and seat changes, cancellations.

Examples:
- A customer disputes a charge or asks for a refund.
- A customer asks what a plan or seat change will cost.
- A customer wants to cancel their subscription.
- A customer asks whether Taskline accepts purchase orders.

Counter-examples:
- Looks like billing because it says "invoice," but is technical, because the customer is reporting that an in-app automation or dashboard widget that happens to be named after invoices or billing is calculating something wrong. The customer's own invoice or charge is not in question; a broken feature is.
- Looks like billing because the customer got locked out while updating their card, but is account, because the thing that needs fixing is the login lockout, not the card.

## Technical

Definition: bugs, errors, outages, performance, integrations, unexpected behavior of any feature, including unexpected logouts or sessions dropping.

Examples:
- A feature throws an error or fails to work as documented.
- An integration (Slack, Google Drive, Jira) stops syncing or duplicates data.
- The app is slow, crashes, or is down for some or all users.
- A user is unexpectedly logged out or a session drops while they are actively using the app.

Counter-examples:
- Looks like technical because the customer says "bug," but is billing, because what they are actually describing is a duplicate charge or a refund need. Customers call billing problems "bugs" all the time; that word does not decide the label.
- Looks like technical because SSO is mentioned, but is account, because the customer is asking how to set SSO up, not reporting that it is broken.

## Account

Definition: login and password problems, two factor auth, SSO setup and configuration, invites, roles and permissions, ownership transfer, profile details, account deletion.

Examples:
- A customer cannot log in, reset a password, or receive a 2FA code.
- A customer wants to invite teammates, change someone's role, or transfer workspace ownership.
- A customer wants to update their profile or delete their account.
- A customer wants to enable or configure SSO for their workspace.

Counter-examples:
- Looks like account because the customer mentions their profile, but is technical, because the actual complaint is that an upload (a photo, an avatar) is failing, which is a feature malfunction, not a request to change a profile detail.
- Looks like account because a session or login is mentioned, but is technical, because the customer is describing repeated unexpected logouts during active use, which the definition above places with session dropping under technical rather than with login problems under account.

## General

Definition: everything else, including feedback, feature requests, sales questions, and partnership requests.

Examples:
- A customer suggests a new feature or shares praise or criticism.
- A prospective customer asks about a demo, a nonprofit discount, or Enterprise features before buying.
- Someone proposes a partnership, an affiliate program, a webinar, or a press interview.

Counter-examples:
- Looks like billing because a discount or price is mentioned, but is general, because there is no existing charge or plan at stake, only a sales or partnership conversation.
- Looks like account because "account executive" is mentioned, but is general, because the customer wants a salesperson, not account team help.

## Tie-break rules for multi-issue emails

1. **The issue the customer most needs resolved wins.** If a customer raises two things but treats one as minor, secondary, or "not urgent," the other one is the label, regardless of order or word count.
2. **If the two issues are genuinely equal in weight, the explicit question wins over background context.** A sentence that explains why the customer is writing (background) is not the same as a sentence that asks something with a question mark (the actual ask). When it is a close call, label by what is being asked, not by what is being explained.

## Decided gray areas

| Situation | Label | Reason |
|---|---|---|
| Customer asks to cancel their subscription | billing | A cancellation is a plan change that stops future charges. It is not an account closure request unless the customer also asks to delete their account or data. |
| Customer asks whether purchase orders are accepted | billing | Purchase orders are a payment method, listed explicitly in the billing definition. |
| Customer asks how to set up or enable SSO | account | Configuring an authentication method for a workspace is account team work. |
| Customer reports that SSO logins are failing or that an SSO-dependent login flow is broken for some or all users | technical | Once SSO is already configured, a failure to authenticate is an outage or bug, not a setup task. |
| Customer reports getting logged out repeatedly or a session dropping during active use | technical | The technical definition explicitly names unexpected logouts and sessions dropping. This is different from not being able to log in at all. |
| Customer cannot log in or gets a persistent "session expired" error immediately at login, with no prior successful session | account | This is a login problem, not a session that dropped mid-use. |
| Customer asks "how do I invite someone" or "how do I set someone's role/permissions" | account | Invites, roles, and permissions are named explicitly in the account definition, even though the plain phrasing can look like a general how-to question. |
| Customer asks to transfer workspace ownership and also asks an explicit question about billing (for example, whether invoices will follow the new owner) | billing | The ownership transfer is often stated as background ("I'm leaving the company"), not asked as a question. When the only explicit question is about billing, the tie-break rule sends the label there. If the customer instead explicitly asks how to transfer ownership, that part is account. |
| Customer describes a duplicate or incorrect charge and calls it a "bug" | billing | The customer's word choice does not change what the ticket is about. A duplicate charge is a billing problem no matter what it is called. |

## How to change this guide

1. Update this guide first. Do not relabel cases before the rule they depend on is written down.
2. Relabel every case affected by the change, including cases beyond the ones that first triggered the change.
3. Bump the dataset's version number and add a changelog entry describing what changed and how many cases were affected.
4. Keep the "Decided gray areas" table current. If a new gray area comes up, add a row for it instead of resolving it silently in one case's notes.
