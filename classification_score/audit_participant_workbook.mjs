import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
import fs from "node:fs/promises";
import path from "node:path";

const workbookPath = process.argv[2];
if (!workbookPath) {
  throw new Error("Usage: node audit_participant_workbook.mjs <participants.xlsx>");
}

const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const summary = await workbook.inspect({
  kind: "workbook,sheet,table,region",
  maxChars: 12000,
  tableMaxRows: 8,
  tableMaxCols: 20,
  tableMaxCellChars: 100,
});
console.log(summary.ndjson);

if (process.argv[3] && process.argv[4]) {
  const [healthyEntries, patientEntries] = await Promise.all([
    fs.readdir(process.argv[3], { withFileTypes: true }),
    fs.readdir(process.argv[4], { withFileTypes: true }),
  ]);
  const healthy = new Set(healthyEntries.filter((x) => x.isDirectory()).map((x) => x.name));
  const patient = new Set(patientEntries.filter((x) => x.isDirectory()).map((x) => x.name));
  const sheet = workbook.worksheets.getItemAt(0);
  const rows = sheet.getUsedRange(true).values;
  const header = rows[0].map(String);
  const col = Object.fromEntries(header.map((name, index) => [name, index]));
  const selected = rows.slice(1).filter((row) => healthy.has(String(row[col.TDBRAIN_ID])) || patient.has(String(row[col.TDBRAIN_ID])));
  const stats = {};
  for (const [name, ids] of [["Healthy", healthy], ["Patient", patient]]) {
    const groupRows = selected.filter((row) => ids.has(String(row[col.TDBRAIN_ID])));
    const ages = groupRows.map((row) => Number(row[col.age])).filter(Number.isFinite);
    const counts = (column) => Object.fromEntries(
      [...groupRows.reduce((acc, row) => {
        const value = String(row[col[column]]);
        acc.set(value, (acc.get(value) ?? 0) + 1);
        return acc;
      }, new Map()).entries()].sort()
    );
    const mean = ages.reduce((a, b) => a + b, 0) / ages.length;
    const variance = ages.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(ages.length - 1, 1);
    stats[name] = {
      requested_subjects: ids.size,
      matched_rows: groupRows.length,
      age_n: ages.length,
      age_mean: mean,
      age_sd: Math.sqrt(variance),
      age_min: Math.min(...ages),
      age_max: Math.max(...ages),
      gender_counts: counts("gender"),
      discovery_replication: counts("DISC/REP"),
      session_season: counts("sessSeason"),
      session_time: counts("sessTime"),
    };
  }
  console.log(JSON.stringify({ selected_cohort_summary: stats }, null, 2));
  if (process.argv[5]) {
    const fields = ["TDBRAIN_ID", "age", "gender", "DISC/REP", "sessSeason", "sessTime"];
    const csvCell = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
    const outputRows = [["subject_id", "group", ...fields.slice(1)]];
    for (const row of selected) {
      const subjectId = String(row[col.TDBRAIN_ID]);
      const group = healthy.has(subjectId) ? "Healthy" : "Patient";
      outputRows.push([subjectId, group, ...fields.slice(1).map((field) => row[col[field]])]);
    }
    await fs.mkdir(path.dirname(process.argv[5]), { recursive: true });
    await fs.writeFile(process.argv[5], outputRows.map((row) => row.map(csvCell).join(",")).join("\n"));
    console.log(JSON.stringify({ metadata_csv: process.argv[5], rows: outputRows.length - 1 }));
  }
}
