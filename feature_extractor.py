import pandas as pd
import url_features as urlfe
import tldextract
import re
from urllib.parse import urlparse

# Function to extract URL-based features
def extract_url_features(url):
    extracted_domain = tldextract.extract(url)
    domain = extracted_domain.domain + '.' + extracted_domain.suffix
    subdomain = extracted_domain.subdomain
    path = urlparse(url).path
    scheme = urlparse(url).scheme
    words_raw = re.split("\-|\.|\/|\?|\=|\@|\&|\%|\:|\_", url.lower())
    
    features = [
        urlfe.url_length(url),
        urlfe.url_length(domain),
        urlfe.having_ip_address(url),
        urlfe.count_dots(url),
        urlfe.count_hyphens(url),
        urlfe.count_at(url),
        urlfe.count_exclamation(url),
        urlfe.count_and(url),
        urlfe.count_or(url),
        urlfe.count_equal(url),
        urlfe.count_underscore(url),
        urlfe.count_tilde(url),
        urlfe.count_percentage(url),
        urlfe.count_slash(url),
        urlfe.count_star(url),
        urlfe.count_colon(url),
        urlfe.count_comma(url),
        urlfe.count_semicolumn(url),
        urlfe.count_dollar(url),
        urlfe.count_space(url),
        urlfe.check_www(words_raw),
        urlfe.check_com(words_raw),
        urlfe.count_double_slash(url),
        urlfe.count_http_token(path),
        urlfe.https_token(scheme),
        urlfe.ratio_digits(url),
        urlfe.ratio_digits(domain),
        urlfe.punycode(url),
        urlfe.port(url),
        urlfe.tld_in_path(extracted_domain.suffix, path),
        urlfe.tld_in_subdomain(extracted_domain.suffix, subdomain),
        urlfe.abnormal_subdomain(url),
        urlfe.count_subdomain(url),
        urlfe.prefix_suffix(url),
        urlfe.shortening_service(url),
        urlfe.path_extension(path),
        urlfe.phish_hints(url),
        urlfe.domain_in_brand(extracted_domain.domain),
        urlfe.brand_in_path(extracted_domain.domain, subdomain),
        urlfe.brand_in_path(extracted_domain.domain, path),
        urlfe.suspecious_tld(extracted_domain.suffix),
    ]
    return features

# Load dataset
input_file = "dataset.xlsx"
output_file = "feature_dataset.xlsx"
df = pd.read_excel(input_file)

if "url" not in df.columns or "label" not in df.columns:
    raise ValueError("Input file must contain 'URL' and 'Label' columns")

# Extract features
feature_names = [
    "length_url", "length_hostname", "ip", "nb_dots", "nb_hyphens", "nb_at", "nb_qm", "nb_and", "nb_or",
    "nb_eq", "nb_underscore", "nb_tilde", "nb_percent", "nb_slash", "nb_star", "nb_colon", "nb_comma",
    "nb_semicolumn", "nb_dollar", "nb_space", "nb_www", "nb_com", "nb_dslash", "http_in_path", "https_token",
    "ratio_digits_url", "ratio_digits_host", "punycode", "port", "tld_in_path", "tld_in_subdomain",
    "abnormal_subdomain", "nb_subdomains", "prefix_suffix", "shortening_service", "path_extension",
    "phish_hints", "domain_in_brand", "brand_in_subdomain", "brand_in_path", "suspecious_tld"
]

feature_data = []
for _, row in df.iterrows():
    url = row["url"]
    label = row["label"]
    features = extract_url_features(url)
    feature_data.append([url] + features + [label])

# Save extracted features
output_df = pd.DataFrame(feature_data, columns=["url"] + feature_names + ["label"])
output_df.to_excel(output_file, index=False)
print(f"Feature extraction completed. Saved to {output_file}")
