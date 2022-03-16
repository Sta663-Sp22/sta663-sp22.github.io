cols = names(openintro::loan50)[ names(openintro::loan50) %in% names(openintro::loans_full_schema) ]

loans = openintro::loans_full_schema[,cols]

readr::write_csv(na.omit(loans), here::here("data/openintro_loans.csv"))
