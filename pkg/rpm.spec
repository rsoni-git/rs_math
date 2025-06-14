%{!?version: %define version 0.0.0}
%{!?release: %define release 0}

Name:           rs_math
Version:        %{version}
Release:        %{release}.%{?dist}
Summary:        A library for computational mathematics.
License:        Proprietary


%description
A library for computational mathematics.


%prep
rm -rf %{_builddir}/%{name}/*
mkdir -p %{_builddir}/%{name}
cp -r %{_sourcedir}/* %{_builddir}/%{name}


%build
cd %{_builddir}/%{name}
cargo test -- --nocapture
cargo build --release --verbose


%install
mkdir -p %{buildroot}/usr/local/lib64/
install -m755 %{_builddir}/%{name}/target/release/librs_math.rlib %{buildroot}/usr/local/lib64/


%files
/usr/local/lib64/librs_math.rlib
