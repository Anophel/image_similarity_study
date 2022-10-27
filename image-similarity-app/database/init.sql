CREATE SEQUENCE default_sequence START 1;


CREATE TABLE users (
   ID INT PRIMARY KEY     NOT NULL,
   name           TEXT    NOT NULL,
   password TEXT    NOT NULL,
   taken boolean default(false)
);

create table triplets (
	ID INT PRIMARY KEY     NOT NULL,
   	target_path           TEXT    NOT NULL,
   	option_one_path           TEXT    NOT NULL,
   	option_two_path           TEXT    NOT null,
   	model_name	text not null,
   	model_favorite int not null,
   	deep_learning_favorite int not null,
   	color_favorite int not null,
   	vlad_favorite int not null
);

create table triplets_annotation (
	ID INT PRIMARY KEY     NOT NULL,
	triplet_id INT REFERENCES triplets(id) not null,
	user_id INT REFERENCES users(id) not null,
	choice INT not null,
	choice_path text not null
);


insert into users values (nextval('default_sequence'), 'admin', 'ChangeMe');
