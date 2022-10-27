# Image similarity app

Web application for triplet annotation.

## Requirements

NodeJS >=16
NPM >=8
PostgreSQL >=12

## Prepare database

```
psql -U username -d databaseName -a -f database/init.sql
psql -U username -d databaseName -a -f database/migration/V01_added_time_annotation_user_info.sql
psql -U username -d databaseName -a -f database/migration/V02_added_triplet_classes.sql
```

## Install

```npm install```

### Run

```npm run start```
