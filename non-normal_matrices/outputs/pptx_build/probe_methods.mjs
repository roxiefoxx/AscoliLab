import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const sourcePptx = "C:/Users/ranic/OneDrive - George Mason University - O365 Production/04_2026 Spring/BINF751 - Biochem Modeling/Final Project/presntation/BINF751_finalproject_RVA_v8.pptx";
const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePptx));
console.log("slides keys", Object.getOwnPropertyNames(Object.getPrototypeOf(presentation.slides)).sort());
console.log("slide keys", Object.getOwnPropertyNames(Object.getPrototypeOf(presentation.slides.getItem(0))).sort());
console.log("shapes keys", Object.getOwnPropertyNames(Object.getPrototypeOf(presentation.slides.getItem(0).shapes)).sort());
const shape = presentation.resolve("sh/1ojy10ne");
console.log("shape keys", Object.getOwnPropertyNames(Object.getPrototypeOf(shape)).sort());
console.log("shape proto", shape.toProto());
