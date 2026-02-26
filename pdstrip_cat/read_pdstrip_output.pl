#!/usr/bin/perl -w
use warnings;
use strict;

my $wave_freq = 0;
my $enc_freq = 0;
my $lambda = 0;
my $wave_angle = 0;
my $speed = 0;
my @translations = {};
my @rotations = {};
my @drifts = {};

my $trans = 0;
my $rot = 0;
my $drift = 0;
my $yaw_drift = 0;
my $surf_riding = 0;
my $found_surf_riding = 0;

my $file = "pdstrip.out";
open my $fh, "<", $file;

chomp(my @lines = <$fh>);

print "freq\t", "enc\t", "angle\t", "speed\t", "surge_r\t", "surge_i\t", "sway_r\t", "sway_i\t", "heave_r\t", "heave_i\t", "roll_r\t", "roll_i\t",
    "pitch_r\t", "pitch_i\t", "yaw_r\t", "yaw_i\t", "surge_d\t", "sway_d\t", "yaw_d\n";
for my $line (@lines) {
    ($wave_freq) = ($line =~ m/frequency.+?(\d+\.\d+)/, $wave_freq);
    ($enc_freq) = ($line =~ m/encounter.+?(\d+\.\d+)/, $enc_freq);
#    ($lambda) = ($line =~ m/wave length.+?(\d+\.\d+)/, $lambda);
    ($wave_angle) = ($line =~ /wave angle.+?(-?\d+\.\d+)/, $wave_angle);
    ($speed) = ($line =~ /speed.+?(-?\d+\.\d+)/, $speed);
    ($trans) = ($line =~ /Translation\s+(-?\d.+)$/, $trans);
    ($rot) = ($line =~ /Rotation\/k\s+(-?\d.+)$/, $rot);
    $surf_riding = $line =~ /SURFRIDING/;
    ($drift) = ($line =~ /transverse drift force per wave amplitude squared\s+(-?\d.+)$/, $drift);
    ($yaw_drift) = ($line =~ /Yaw drift moment per wave amplitude squared\s+(-?\d.+)$/, $yaw_drift);

    if ($trans) {
        @translations = split /\s+/, $trans;
    }
    if ($rot) {
        @rotations = split /\s+/, $rot;
    }
    if ($drift) {
        @drifts = split /\s+/, $drift;
    }
    if ($surf_riding) {
        $found_surf_riding = 1;
        $trans = 1;
        $rot = 1;
    }

    if ($wave_freq && $enc_freq  && $wave_angle && $speed && $trans && $rot && $drift && $yaw_drift) {
        if ($found_surf_riding) {
            @translations = (0, 0, 0, 0, 0, 0, 0, 0);
            @rotations = (0, 0, 0, 0, 0, 0, 0, 0);
            @drifts =  (0, 0);
            $yaw_drift = 0.0;
        }
        printf("%6.3f\t%6.3f\t%6.1f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%6.3f\t%8.5e\t%8.5e\t%8.5e\n",
               $wave_freq, $enc_freq, $wave_angle, $speed,
               $translations[0], $translations[1], $translations[3], $translations[4], $translations[6], $translations[7],
               $rotations[0], $rotations[1], $rotations[3], $rotations[4], $rotations[6], $rotations[7], $drifts[0], $drifts[1], $yaw_drift);
        
        $wave_freq = 0;
        $enc_freq = 0;
        $wave_angle = 0;
        $speed = 0;
        $trans = 0;
        $rot = 0;
        $drift = 0;
        $yaw_drift = 0;
        $surf_riding = 0;
        $found_surf_riding = 0;
    }
}
