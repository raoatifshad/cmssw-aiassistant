"""Neo4j connection and schema setup. Standalone — no imports from any other project."""
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "cmssdt2026")


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def setup_schema(driver):
    """Uniqueness constraints + indexes. Idempotent (IF NOT EXISTS)."""
    with driver.session() as s:
        for constraint in s.run("SHOW CONSTRAINTS").data():
            legacy_ib_constraint = (
                constraint.get("type") == "UNIQUENESS"
                and constraint.get("labelsOrTypes") == ["IB"]
                and constraint.get("properties") == ["tag"]
            )
            legacy_warning_constraint = (
                constraint.get("type") == "UNIQUENESS"
                and constraint.get("labelsOrTypes") == ["Warning"]
                and constraint.get("properties") == ["hash"]
            )
            if legacy_ib_constraint or legacy_warning_constraint:
                name = constraint["name"].replace("`", "``")
                s.run(f"DROP CONSTRAINT `{name}` IF EXISTS")

    stmts = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:IB) REQUIRE n.ib_key IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Package) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Checker) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:SourceFile) REQUIRE n.path IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Author) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Warning) REQUIRE n.key IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:RelVal) REQUIRE n.key IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:RelValException) REQUIRE n.key IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:ExitCode) REQUIRE n.code IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Concept) REQUIRE n.concept_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:AlertRule) REQUIRE n.rule_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:RouterShadowLog) REQUIRE n.log_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:UnitTest) REQUIRE n.key IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:PRComment) REQUIRE n.key IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:GitHubComment) REQUIRE n.key IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Commit) REQUIRE n.sha IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:WorkflowCommand) REQUIRE n.key IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Issue) REQUIRE n.issue_key IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Label) REQUIRE n.key IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:ImportState) REQUIRE n.source_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:UserFeedback) REQUIRE n.feedback_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:ChatSession) REQUIRE n.session_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:ChatTurn) REQUIRE n.turn_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:EventOccurrence) REQUIRE n.event_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Build) REQUIRE n.tag IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Service) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Host) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:DocChunk) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:TestLogRun) REQUIRE n.key IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:LogProblem) REQUIRE n.key IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:ProblemSignature) REQUIRE n.signature_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:ProblemFirstSeen) REQUIRE n.key IS UNIQUE",
        "CREATE INDEX IF NOT EXISTS FOR (n:IB) ON (n.tag)",
        "CREATE INDEX IF NOT EXISTS FOR (n:IB) ON (n.release)",
        "CREATE INDEX IF NOT EXISTS FOR (n:IB) ON (n.parent_tag)",
        "CREATE INDEX IF NOT EXISTS FOR (n:PR) ON (n.repo, n.number)",
        "CREATE INDEX IF NOT EXISTS FOR (n:WorkflowCommand) ON (n.workflow_id)",
        "CREATE INDEX IF NOT EXISTS FOR (n:RelVal) ON (n.ib_key)",
        # get_relval_recurrence looks a workflow up across every build; without this it
        # read every RelVal (~950k db hits, 110 ms at 475k rows, 2026-10-15) and grew with them
        "CREATE INDEX IF NOT EXISTS FOR (n:RelVal) ON (n.workflow_id)",
        "CREATE INDEX IF NOT EXISTS FOR (n:RelValException) ON (n.ib_key, n.workflow_id)",
        "CREATE INDEX IF NOT EXISTS FOR (n:UnitTest) ON (n.ib_key)",
        "CREATE INDEX IF NOT EXISTS FOR (n:Warning) ON (n.package)",
        "CREATE INDEX IF NOT EXISTS FOR (n:ChatTurn) ON (n.turn_order)",
        "CREATE INDEX IF NOT EXISTS FOR (n:EventOccurrence) ON (n.timestamp)",
        "CREATE INDEX IF NOT EXISTS FOR (n:EventOccurrence) ON (n.severity)",
        "CREATE INDEX IF NOT EXISTS FOR (n:Concept) ON (n.status, n.expires_at)",
        "CREATE INDEX IF NOT EXISTS FOR (n:UserFeedback) ON (n.status)",
        "CREATE FULLTEXT INDEX github_artifact_text IF NOT EXISTS "
        "FOR (n:Issue|PR) ON EACH [n.title, n.body]",
        "CREATE FULLTEXT INDEX shifterDocText IF NOT EXISTS "
        "FOR (n:DocChunk) ON EACH [n.heading, n.text]",
        # build_graph.py --reconcile deletes by (label, source_id) -- see RECONCILE_LABELS
        "CREATE INDEX IF NOT EXISTS FOR (n:RelVal) ON (n.source_id)",
        "CREATE INDEX IF NOT EXISTS FOR (n:RelValException) ON (n.source_id)",
        "CREATE INDEX IF NOT EXISTS FOR (n:UnitTest) ON (n.source_id)",
        "CREATE INDEX IF NOT EXISTS FOR (n:PR) ON (n.source_id)",
        "CREATE INDEX IF NOT EXISTS FOR (n:WorkflowCommand) ON (n.source_id)",
        "CREATE INDEX IF NOT EXISTS FOR (n:Warning) ON (n.source_id)",
        "CREATE INDEX IF NOT EXISTS FOR (n:TestLogRun) ON (n.source_id)",
        "CREATE INDEX IF NOT EXISTS FOR (n:TestLogRun) ON (n.ib_key)",
        "CREATE INDEX IF NOT EXISTS FOR (n:LogProblem) ON (n.source_id)",
        "CREATE INDEX IF NOT EXISTS FOR (n:LogProblem) ON (n.ib_key)",
        "CREATE INDEX IF NOT EXISTS FOR (n:LogProblem) ON (n.kind, n.in_passed)",
        "CREATE INDEX IF NOT EXISTS FOR (n:ProblemFirstSeen) ON (n.first_seen_when)",
        "CREATE FULLTEXT INDEX logProblemText IF NOT EXISTS "
        "FOR (n:LogProblem) ON EACH [n.summary, n.text]",
        # every specific name and "<category>|<producer>" key of a test-log problem, matched
        # whole (whitespace analyzer: no splitting at '::', '_' or '.', no stemming)
        "CREATE FULLTEXT INDEX logProblemNames IF NOT EXISTS "
        "FOR (n:LogProblem) ON EACH [n.names_text, n.context_keys_text] "
        "OPTIONS {indexConfig: {`fulltext.analyzer`: 'whitespace'}}",
    ]
    with driver.session() as s:
        s.run("MATCH (comment:PRComment) SET comment:GitHubComment")
        s.run(
            "MATCH (ib:IB) WHERE ib.tag =~ '^CMSSW_[0-9]+_[0-9]+(?:_.+)?_X_[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{4}$' "
            "WITH ib, split(ib.tag, '_') AS parts "
            "SET ib.parent_tag = CASE WHEN size(parts) = 5 THEN ib.tag "
            "ELSE parts[0] + '_' + parts[1] + '_' + parts[2] + '_X_' + parts[-1] END, "
            "ib.variant = CASE WHEN size(parts) = 5 THEN NULL "
            "ELSE reduce(name = '', part IN parts[3..-2] | name + CASE WHEN name = '' THEN '' ELSE '_' END + part) END"
        )
        for stmt in stmts:
            try:
                s.run(stmt)
            except Exception as e:
                print(f"  note: {str(e)[:80]}")
    print("Schema constraints set up.")


def clear_graph(driver):
    """Wipe everything — use only for a clean rebuild."""
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    print("Graph cleared.")


if __name__ == "__main__":
    d = get_driver()
    setup_schema(d)
    d.close()
