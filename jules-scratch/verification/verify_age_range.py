from playwright.sync_api import sync_playwright, expect

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    try:
        page.goto("http://localhost:5001/webcam")

        # Click the capture button
        capture_button = page.locator("#captureBtn")
        expect(capture_button).to_be_visible()
        capture_button.click()

        # Wait for the results to be displayed
        results = page.locator("#results")
        expect(results).to_be_visible(timeout=10000)

        # Check for the "Age Range" label
        age_range_label = page.locator("h3:has-text('Age Range')")
        expect(age_range_label).to_be_visible()

        # Take a screenshot
        page.screenshot(path="jules-scratch/verification/verification.png")
        print("Screenshot taken successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        page.screenshot(path="jules-scratch/verification/error.png")

    finally:
        browser.close()

with sync_playwright() as playwright:
    run(playwright)