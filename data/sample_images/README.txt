# Sample Prescription Images

This directory is intended to store sample prescription images for testing the prescription processing system.

## Adding Sample Images

To add a sample prescription image:

1. Place your prescription image files in this directory
2. Use common image formats like JPG, PNG, or TIFF
3. Ensure the images are clear and readable
4. Consider using anonymized or synthetic prescription images to protect privacy

## Sample Image Requirements

For best results, sample images should:
- Have good lighting and contrast
- Be properly oriented (not rotated)
- Have minimal background noise
- Include all relevant prescription information

## Example Usage

Once you've added sample images, you can process them using the main script:

```
python src/main.py --image data/sample_images/your_image.jpg
```

Or use the example script which looks for a file named "sample_prescription.jpg":

```
python src/example.py
```

## Privacy Notice

If using real prescription images:
- Ensure all personal identifiable information is redacted
- Obtain proper permissions if using real medical data
- Consider using synthetic or mock prescription images instead

## Creating Synthetic Prescription Images

If you don't have access to real prescription images, you can create synthetic ones:

1. Use a word processor to create a template prescription
2. Print it out and handwrite the prescription details
3. Scan or photograph the handwritten prescription
4. Save the image in this directory

This approach allows you to test the system without privacy concerns.