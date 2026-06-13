# ATP all-in-AWS Bedrock demo — architecture

Two views of the `infra/` Terraform demo: a resource/IAM map (where processes
run) and the request flow of a single methodology sweep.

## Resource & IAM map

```mermaid
flowchart TB
    subgraph OP["🖥️ Operator machine (your Mac)"]
        TF[terraform apply/destroy]
        POD[podman build + push]
        PRE[preflight.sh]
        RS[run-sweep.sh]
        TUN[ssh -L / ssm port-forward]
    end

    subgraph AWS["☁️ AWS account · eu-central-1"]
        ECR[("ECR repo<br/>atp-platform:latest<br/>lifecycle: keep last 3")]
        BUD["Budgets $5 alert"]

        subgraph EC2["EC2 t3.small · AL2023 · IMDSv2 hop=2"]
            SG["Security group<br/>SSH:22 optional · NEVER 8080"]
            DKR["Docker daemon"]
            DASH["atp-dashboard<br/>127.0.0.1:8080"]
            RUN["atp test container (--rm)"]
            VOL[("atp-data volume<br/>SQLite /root/.atp")]
            SSM["SSM agent"]
        end

        subgraph IAMG["IAM — no static keys"]
            ROLE["EC2 instance role<br/>InvokeAgent + InvokeModel*<br/>+ GetInferenceProfile<br/>+ ECR pull + SSM core"]
            AROLE["Bedrock agent exec-role<br/>InvokeModel* + GetInferenceProfile"]
        end

        subgraph BR["Bedrock — managed (no server you run)"]
            AGENT["Agent under test<br/>agent_id · alias 'live'"]
            MODEL["Claude Sonnet 4.5<br/>EU cross-region<br/>inference profile"]
        end
    end

    TF -->|provisions| AWS
    POD -->|push image| ECR
    PRE -.->|"your SSO/profile creds:<br/>describe + 1-token invoke"| BR
    RS -->|"ssh:22 / ssm send-command"| EC2
    TUN -->|8080 over tunnel| DASH

    ECR -->|pull| DKR
    DKR --> DASH
    DKR --> RUN
    RUN -->|reads creds via IMDSv2| ROLE
    ROLE -->|"bedrock:InvokeAgent"| AGENT
    ROLE -->|"bedrock:InvokeModel<br/>the judge"| MODEL
    AGENT -. assumes .-> AROLE
    AROLE -->|orchestration model| MODEL
    RUN -->|write results| VOL
    DASH -->|read results| VOL

    classDef managed fill:#fff3cd,stroke:#d39e00;
    classDef box fill:#d4edda,stroke:#28a745;
    classDef op fill:#cce5ff,stroke:#0066cc;
    class AGENT,MODEL,ECR,BUD managed;
    class DKR,DASH,RUN,VOL,SSM,SG box;
    class TF,POD,PRE,RS,TUN op;
```

## Single-run sequence

```mermaid
sequenceDiagram
    autonumber
    actor Op as Operator (your Mac)
    participant EC2 as EC2 box<br/>(Docker daemon)
    participant IMDS as IMDSv2<br/>(instance role, hop=2)
    participant ATP as atp test container<br/>(--rm, --adapter=bedrock)
    participant Agent as Bedrock Agent<br/>(under test, alias)
    participant Judge as Bedrock model<br/>(Sonnet 4.5 judge,<br/>EU inference profile)
    participant Vol as atp-data volume<br/>(SQLite /root/.atp)
    participant Dash as atp-dashboard<br/>(127.0.0.1:8080)

    Op->>EC2: run-sweep.sh → ssh/ssm:<br/>sudo docker run … atp test … --runs=3
    EC2->>ATP: start container (mounts atp-data)
    ATP->>IMDS: read instance role creds (boto3 default chain)
    IMDS-->>ATP: temporary IAM credentials

    loop 4 cases × 3 runs
        ATP->>Agent: bedrock:InvokeAgent (policy excerpt)
        Note over Agent: orchestrates via its<br/>foundationModel (EU profile)
        Agent-->>ATP: response + traces (datetime fields)
        ATP->>Judge: bedrock:InvokeModel (critical_check + rubric)
        Judge-->>ATP: verdict + score (0..1)
        Note over ATP: hard-gate: fabricated deadline → 0,<br/>else weighted rubric
        ATP->>Vol: persist result + artifact<br/>(_json_safe coerces datetime → ISO)
    end

    ATP-->>EC2: exit (container removed, --rm)
    EC2-->>Op: sweep summary (stdout / SSM)

    Note over Vol,Dash: dashboard mounts the SAME named volume → same SQLite DB
    Op->>Dash: ssh -L 8080 tunnel → GET /ui/runs
    Dash->>Vol: query results
    Vol-->>Dash: Execution #1 (clean/moderate/very_severe=80, severe=0)
    Dash-->>Op: rendered run detail
```
