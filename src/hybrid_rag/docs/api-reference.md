# API: limits, errors and webhooks

## Authentication

The REST API is available on Team, Business and Enterprise. Requests are authenticated with a personal
access token in the Authorization header (Bearer). Tokens are created under Profile > API tokens, expire
after 12 months and can be revoked at any time. A token has the same permissions as the user who made it.

## Rate limits

The API allows 60 requests per minute per token on the Team plan and 300 requests per minute on Business.
Enterprise limits are agreed in the contract, by default 1,000 requests per minute. When the limit is
exceeded the API answers with HTTP 429 and error code TB-4290; the Retry-After header says how many
seconds to wait.

## Error codes

TB-4010: the token is missing, expired or revoked. TB-4030: the user has no permission for this board.
TB-4031: the board is archived and read-only; unarchive it before writing to it. TB-4040: the board,
card or list does not exist. TB-4220: the request body failed validation; the response lists the
invalid fields. TB-5000: internal error; retry later and contact support with the request ID if it
persists.

## Pagination

List endpoints return at most 100 items per page. The response contains a next_cursor field; pass it as
the cursor parameter to get the next page. An empty next_cursor means there are no more pages.

## Webhooks

Webhooks send a POST request to your URL when cards are created, moved or completed. Each request is
signed with HMAC-SHA256 in the X-Talvi-Signature header. If your endpoint does not answer with a 2xx status
within 10 seconds, we retry up to 5 times with exponential backoff (1, 5, 15, 60 and 240 minutes). After
5 failed retries the webhook is paused and the workspace admins get an email.

## API versioning

The current API version is v2. Breaking changes only happen in a new major version, and an old version is
supported for at least 12 months after a successor is released. API v1 will be switched off on
30 June 2027.
