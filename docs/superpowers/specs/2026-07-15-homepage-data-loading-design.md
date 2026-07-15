# Homepage Data Loading Design

## Goal

The homepage must request only data needed by visible homepage controls. Image records and thumbnail files must not load until the user opens the image management page.

## Design

Add a lightweight `GET /api/dashboard-stats` endpoint that returns only the homepage values for today's generated-image count and detection success rate. The existing `GET /api/images` endpoint remains responsible for image records and image-management statistics.

During frontend initialization, call a new dashboard-stat loader instead of `loadImages()`. Keep `loadImages()` behind the image-management navigation path and existing operations that need to refresh image records after a successful mutation.

This separates the data flow as follows:

- Homepage initialization: `/api/dashboard-stats` only.
- First and subsequent visits to image management: `/api/images`, followed by table rendering and thumbnail requests for the visible page.
- Successful image creation, deletion, or trace operations: preserve the existing record refresh behavior where it is needed.

If the dashboard-stat request fails, the existing default values remain visible and other homepage functionality continues to work.

## Testing

Keep regression coverage small:

- Verify the dashboard-stat endpoint returns the required values without an image list.
- Verify frontend initialization loads dashboard stats and does not directly call `loadImages()`.

Run the focused tests plus the existing lightweight frontend contract tests affected by `index.html`.
