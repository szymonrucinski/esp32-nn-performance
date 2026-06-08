#pragma once
// EuroSAT land-cover classes, ALPHABETICAL order == training label index
// (torchvision ImageFolder sorts class dirs alphabetically). Order must match
// the model's output head exactly or every prediction is mislabeled.
static const char *eurosat_cat_names[] = {
    "AnnualCrop",            // 0
    "Forest",               // 1
    "HerbaceousVegetation", // 2
    "Highway",              // 3
    "Industrial",           // 4
    "Pasture",              // 5
    "PermanentCrop",        // 6
    "Residential",          // 7
    "River",                // 8
    "SeaLake",              // 9
};
