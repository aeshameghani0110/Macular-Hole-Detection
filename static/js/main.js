FilePond.registerPlugin(FilePondPluginImagePreview);

// Initialize FilePond
const pond = FilePond.create(document.querySelector("#image-upload"), {
    allowImagePreview: true,
    imagePreviewHeight: 200,
    allowMultiple: false,
    acceptedFileTypes: ["image/*"],
    fileValidateTypeLabelExpectedTypes: "Upload a valid image file",
});