
alter table triplets_annotation add column time_spent int default 0;

alter table users add column email text, add column age_group text, add column education text, add column ml_expert text;
