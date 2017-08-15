/*
 * differential.hpp
 * 
 * Defines a type FirstDifferential, that can replace a double in
 * computations and calculates derivatives along with the main value.
 *
 * For example, the code
 *
 *   double a = 2;
 *   double b = 3;
 *   double x = 5 * a + pow(b, 2);
 *
 * would yield x = 13. If we want to know the derivatives of x with respect to
 * a and b, we can use the FirstDifferential class to calculate them:
 *   
 *   // Shortcut to a FirstDifferential object:
 *   typedef FirstDifferential Diff
 *   
 *   // Create a and b as differentials. Both have derivative 1 with respect to
 *   // themselves. This can be indicated explicitly or, as in this example,
 *   // using the constructor arguments (value, indice) where indice is the
 *   // indice of this variable in the list of derivatives.
 *   Diff a = Diff(2, 0);
 *   Diff b = Diff(3, 1);
 *   Diff x = 5 * a + pow(b, 2);
 *   
 * Now x->value or (double)x will return 13, while x[0] = 5 and x[1] = b.
 *
 * N.B.: The functions fabs, floor, ceil and fmod are implemented, but will
 * yield "nan" for the derivatives.
 * 
 * This file is part of Myokit
 *  Copyright 2011-2017 Maastricht University
 *  Licensed under the GNU General Public License v3.0
 *  See: http://myokit.org
 *
 * Authors:
 *  Michael Clerx
 *
 * Adapted from myokit_differential.h, copyright Pieter Collins.
 * Original copyright message posted below:
 *
 *  Copyright 2013 Pieter Collins
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Library General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 */

#include <math.h>
#include <assert.h>

/*
 * Requires two macros to be set:
 *  N_DIFFS     Specifying the size of each differential
 *  Real        Specifying a real number type (i.e. double)        
 */

/*
 * Covector. A tiny, fixed size array class that allows addition and scalar
 * multiplication.
 */
class Covector {
    Real data[N_DIFFS];
public:
    /*
     * Create an empty covector.
     */
    Covector()
    {
        for(size_t i=0; i<N_DIFFS; i++) this->data[i] = 0;
    }
    
    /*
     * Access or change an element of the covector.
     */
    Real& operator[] (size_t i) { return this->data[i]; }
    Real const& operator[] (size_t i) const { return this->data[i]; }
    
    /*
     * Comparison operators
     */
    bool operator==(const Covector &other) const
    {
        const Real* a = this->data;
        const Real* b = other.data;
        for(size_t i=0; i<N_DIFFS; i++) {
            if(*a != *b) return 0;
            a++; b++;
        }
        return 1;
    }
    bool operator!=(const Covector &other) const
    {
        const Real* a = this->data;
        const Real* b = other.data;
        for(size_t i=0; i<N_DIFFS; i++) {
            if(*a != *b) return 1;
            a++; b++;
        }
        return 0;
    }

    /*
     * Addition with covectors
     */
    void operator+=(const Covector &other)
    { 
        for(size_t i=0; i<N_DIFFS; i++) this->data[i] += other.data[i];
    }
    void operator-=(const Covector &other)
    { 
        for(size_t i=0; i<N_DIFFS; i++) this->data[i] -= other.data[i];
    }
    Covector operator+(const Covector &x) const
    {
        Covector z = Covector();
        for(size_t i=0; i<N_DIFFS; i++) z.data[i] = this->data[i] + x.data[i];
        return z;
    }
    Covector operator-(const Covector &x) const
    {
        Covector z = Covector();
        for(size_t i=0; i<N_DIFFS; i++) z.data[i] = this->data[i] - x.data[i];
        return z;
    }
    Covector operator+() const
    {
        Covector z = Covector();
        for(size_t i=0; i<N_DIFFS; i++) z.data[i] = this->data[i];
        return z;
    }
    Covector operator-() const
    {
        Covector z = Covector();
        for(size_t i=0; i<N_DIFFS; i++) z.data[i] = -this->data[i];
        return z;
    }
    
    /*
     * Scalar multiplication. (Operation Real * Covector is implemented
     * externally).
     */
    void operator*=(const Real x)
    { 
        for(size_t i=0; i<N_DIFFS; i++) this->data[i] *= x;
    }
    void operator/=(Real x)
    { 
        x = 1./x;
        for(size_t i=0; i<N_DIFFS; i++) this->data[i] *= x;
    }
    Covector operator*(const Real x) const
    {
        Covector z = Covector();
        for(size_t i=0; i<N_DIFFS; i++) z.data[i] = this->data[i] * x;
        return z;
    }
    Covector operator/(Real x) const
    {
        x = 1./x;
        Covector z = Covector();
        for(size_t i=0; i<N_DIFFS; i++) z.data[i] = this->data[i] * x;
        return z;
    }
    
    /*
     * Reciprocal of a Covector
     */
    Covector rec() const
    {
        Covector z = Covector();
        for(size_t i=0; i<N_DIFFS; i++) z.data[i] = 1. / this->data[i];
        return z;
    }   
};

/*
 * Multipliciation of a scalar by a covector.
 */ 
Covector operator*(const Real s, const Covector &v)
{
    return v * s;
}
Covector operator/(const Real s, const Covector &v)
{
    Covector x = v.rec();
    x *= s;
    return x;
}
    
/*
 * Numeric type for automatic integration: stores a value and a list of
 * derivatives.
 */
class FirstDifferential
{
    private:

    Real v;
    Covector d;

    public:

    /*
     * Constructs a differential.
     */
    explicit FirstDifferential() : v(), d()
    {
        // Value is fine, gradient automatically fills itself with zeros.
    }
    
    /* 
     * Constructs a differential with a value.
     * This is the only constructor allowed not to be explicit!
     */
    FirstDifferential(const Real v) : v(v), d() {}

    /*
     * Constructs a differential with value v and gradient g.
     */
    explicit FirstDifferential(const Real v, const Covector g)
        : v(v), d(g) {}
    
    /*
     * Constructs a differential with value v, and an all-zero derivative
     * vector except for a 1 at position i.
     */
    explicit FirstDifferential(const Real v, const int i)
        : v(v), d()
    {
        assert(0 <= i);
        assert(i < N_DIFFS);
        this->d[i] = 1;
    }       

    /*
     * Read-access to the values of this differential.
     */
    const Real& value() const
    {
        return this->v;
    }

    const Covector& derivatives() const
    {
        return this->d;
    }
     
    const Real& operator[](size_t i) const
    {
        return this->d[i];
    }
    
    /*
     * Write access
     */
    void value(const Real v)
    {
        this->v = v;
    }
    
    Real& operator[](const size_t i)
    {
        // Returning a reference to this value allows a new value to be assigned!
        return this->d[i];
    }
    
    /*
     * Comparison operators
     */
    bool operator==(const FirstDifferential &other) const
    {
        return this->v == other.v && this->d == other.d;
    }
    bool operator!=(const FirstDifferential &other) const
    {
        return this->v != other.v || this->d != other.d;
    }
    bool operator>(const FirstDifferential &other) const
    {
        return this->v > other.v;
    }
    bool operator<(const FirstDifferential &other) const
    {
        return this->v < other.v;
    }
    bool operator>=(const FirstDifferential &other) const
    {
        return this->v >= other.v;
    }
    bool operator<=(const FirstDifferential &other) const
    {
        return this->v <= other.v;
    }

    /*
     * Comparison with reals
     */
    bool operator==(const Real y) const
    {
        return this->v == y;
    }
    bool operator!=(const Real y) const
    {
        return this->v != y;
    }
    bool operator>(const Real y) const
    {
        return this->v > y;
    }
    bool operator<(const Real y) const
    {
        return this->v < y;
    }
    bool operator>=(const Real y) const
    {
        return this->v >= y;
    }
    bool operator<=(const Real y) const
    {
        return this->v <= y;
    }
    
    /*
     * Addition
     */
    void operator+=(const FirstDifferential &x)
    {
        this->v += x.v;
        this->d += x.d;
    }
    void operator-=(const FirstDifferential &x)
    {
        this->v -= x.v;
        this->d -= x.d;
    }
    void operator+=(const Real c)
    {
        this->v += c;
    }
    void operator-=(const Real c)
    {
        this->v -= c;
    }
    FirstDifferential operator+() const
    {
        return FirstDifferential(this->v, this->d);
    }
    FirstDifferential operator-() const
    {
        return FirstDifferential(-this->v, -this->d);
    }
    FirstDifferential operator+(const FirstDifferential &x) const
    {
        FirstDifferential z = FirstDifferential(this->v, this->d);
        z += x;
        return z;
    }
    FirstDifferential operator-(const FirstDifferential &x) const
    {
        FirstDifferential z = FirstDifferential(this->v, this->d);
        z -= x;
        return z;
    }
    FirstDifferential operator+(const Real x) const
    {
        FirstDifferential z = FirstDifferential(this->v, this->d);
        z += x;
        return z;
    }
    FirstDifferential operator-(const Real x) const
    {
        FirstDifferential z = FirstDifferential(this->v, this->d);
        z -= x;
        return z;
    }
    
    /*
     * Multiplication
     */
    void operator*=(const FirstDifferential &x)
    {
        this->d *= x.v;
        this->d += this->v * x.d;
        this->v *= x.v;
    }
    void operator/=(const FirstDifferential &x)
    {
        this->v /= x.v;
        this->d -= this->v * x.d;
        this->d /= x.v;
    }
    void operator*=(const Real c)
    {
        this->v *= c;
        this->d *= c;
    }
    void operator/=(const Real c)
    {
        this->v /= c;
        this->d /= c;
    }
    FirstDifferential operator*(const FirstDifferential &x) const
    {
        FirstDifferential z = FirstDifferential(this->v, this->d);
        z *= x;
        return z;
    }
    FirstDifferential operator/(const FirstDifferential &x) const
    {
        FirstDifferential z = FirstDifferential(this->v, this->d);
        z /= x;
        return z;
    }
    FirstDifferential operator*(const Real x) const
    {
        FirstDifferential z = FirstDifferential(this->v, this->d);
        z *= x;
        return z;
    }
    FirstDifferential operator/(const Real x) const
    {
        FirstDifferential z = FirstDifferential(this->v, this->d);
        z /= x;
        return z;
    }
    
    /*
     * Reciprocal of a first differential
     */
    FirstDifferential rec() const
    {
        return FirstDifferential(1. / v, -d / (v * v));
    }
};

/*
 * Operations on a (Real, FirstDifferential) pair.
 */
FirstDifferential
operator+(const Real c, const FirstDifferential &x)
{
    return x+c;
}
FirstDifferential
operator-(const Real c, const FirstDifferential &x)
{
    return (-x)+c;
}
FirstDifferential
operator*(const Real c, const FirstDifferential &x)
{
    return x*c;
}
FirstDifferential
operator/(const Real c, const FirstDifferential &x)
{
    return x.rec()*c;
}

/*
 * Functions
 */
FirstDifferential pow(const FirstDifferential &x, const Real r)
{
    return FirstDifferential(pow(x.value(), r), (r * pow(x.value(), r - 1)) * x.derivatives());
}
FirstDifferential pow(const Real r, const FirstDifferential &x)
{
    Real p = pow(r, x.value());
    return FirstDifferential(p, p * log(r) * x.derivatives());
}
FirstDifferential pow(const FirstDifferential &x, const FirstDifferential &y)
{
    Real p = pow(x.value(), y.value());
    return FirstDifferential(p, p * (log(x.value()) * y.derivatives() + y.value() / x.value() * x.derivatives()));
    //return exp(log(x)*y);
}
FirstDifferential sqrt(const FirstDifferential &x)
{
    Real sqrt_val = sqrt(x.value());
    return FirstDifferential( sqrt_val, 1./(2*sqrt_val)*x.derivatives() );
}
FirstDifferential exp(const FirstDifferential &x)
{
    Real exp_val = exp(x.value());
    return FirstDifferential( exp_val, exp_val*x.derivatives() );
}
FirstDifferential log(const FirstDifferential &x)
{
    Real log_val = log(x.value());
    Real rec_val = 1./x.value();
    return FirstDifferential( log_val, rec_val*x.derivatives() );
}
FirstDifferential sin(const FirstDifferential &x)
{
    Real sin_val = sin(x.value());
    Real cos_val = cos(x.value());
    return FirstDifferential( sin_val, cos_val*x.derivatives() );
}
FirstDifferential cos(const FirstDifferential &x)
{
    Real cos_val = cos(x.value());
    Real neg_sin_val = -sin(x.value());
    return FirstDifferential( cos_val, neg_sin_val*x.derivatives() );
}
FirstDifferential tan(const FirstDifferential &x)
{
    Real tan_val = tan(x.value());
    Real cos_val = cos(x.value());
    Real sqr_sec_val = 1./(cos_val*cos_val);
    return FirstDifferential( tan_val, sqr_sec_val*x.derivatives() );
}
FirstDifferential asin(const FirstDifferential &x)
{
    Real asin_val = asin(x.value());
    Real d_asin_val = 1./(sqrt(1-x.value()*x.value()));
    return FirstDifferential( asin_val, d_asin_val*x.derivatives() );
}
FirstDifferential acos(const FirstDifferential &x)
{
    Real acos_val = acos(x.value());
    Real d_acos_val = -1./(sqrt(1-x.value() * x.value()));
    return FirstDifferential( acos_val, d_acos_val*x.derivatives() );
}
FirstDifferential atan(const FirstDifferential &x)
{
    Real atan_val = atan(x.value());
    Real d_atan_val = 1./(1+x.value()*x.value());
    return FirstDifferential( atan_val, d_atan_val*x.derivatives() );
}
FirstDifferential fabs(const FirstDifferential &x)
{
    return FirstDifferential(fabs(x.value()), NAN * x.derivatives());
}
FirstDifferential floor(const FirstDifferential &x)
{
    return FirstDifferential(floor(x.value()), NAN * x.derivatives());
}
FirstDifferential ceil(const FirstDifferential &x)
{
    return FirstDifferential(ceil(x.value()), NAN * x.derivatives());
}
FirstDifferential fmod(const FirstDifferential &x, const FirstDifferential &y)
{
    return FirstDifferential(fmod(x.value(), y.value()), NAN * x.derivatives());
}
FirstDifferential fmod(const FirstDifferential &x, const Real &y)
{
    return FirstDifferential(fmod(x.value(), y), NAN * x.derivatives());
}
FirstDifferential fmod(const Real &x, const FirstDifferential &y)
{
    return FirstDifferential(fmod(x, y.value()), NAN * y.derivatives());
}

/*
 * Diff & Real if-then-else functions
 */
FirstDifferential ifte(bool condition, const FirstDifferential& rtrue, const FirstDifferential& rfalse) {
    if(condition) { return rtrue; } else { return rfalse; }
}
FirstDifferential ifte(bool condition, const FirstDifferential& rtrue, const Real rfalse) {
    if(condition) { return rtrue; } else { return FirstDifferential(rfalse); }
}
FirstDifferential ifte(bool condition, const Real rtrue, const FirstDifferential& rfalse) {
    if(condition) { return FirstDifferential(rtrue); } else { return rfalse; }
}
Real ifte(bool condition, const Real rtrue, const Real rfalse) {
    if(condition) { return rtrue; } else { return rfalse; }
}
